"""Single-run training, evaluation, profiling, and report export."""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.metrics import MedicalAnswerNormalizer, classification_metrics, generation_metrics
from ..utils.reporting import (
    REFERENCE_TABLE_II,
    REFERENCE_TABLE_III,
    build_table_ii_row,
    build_table_iii_row,
    dataset_characteristics,
    write_json,
    write_rows_csv,
)
from ..utils.visualizer import OrganSaliencyVisualizer
from .losses import MedVQALossEngine
from .profiler import HardwareDeploymentProfiler


class SingleRunExecutionEngine:
    """Train, validate, test, profile, visualize, and export one experiment."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        answer_vocab: List[str],
        output_dir: str,
        config_mode: str = "m5",
        dataset_name: str = "vqa_rad",
        device: Optional[str] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        objective_schedule: str = "supervised",
        compute_bertscore: bool = False,
        seed: int = 42,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.answer_vocab = [MedicalAnswerNormalizer.normalize(answer) for answer in answer_vocab]
        self.ans2idx = {answer: index for index, answer in enumerate(self.answer_vocab)}
        self.unknown_index = self.ans2idx.get(MedicalAnswerNormalizer.normalize("[UNK]"), 0)
        self.output_dir = output_dir
        self.config_mode = config_mode.lower()
        self.dataset_name = dataset_name.lower().replace("-", "_")
        self.objective_schedule = objective_schedule.lower()
        self.compute_bertscore = compute_bertscore
        self.seed = seed
        if self.objective_schedule not in {"supervised", "stochastic"}:
            raise ValueError("objective_schedule must be 'supervised' or 'stochastic'")
        self.random = random.Random(seed)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_engine = MedVQALossEngine(config_mode=self.config_mode)
        os.makedirs(output_dir, exist_ok=True)

    def execute_single_run(self, epochs: int = 1) -> Dict[str, Any]:
        """Run one experiment and write the report artifacts described in Tables I--VI."""

        best_val_acc = -1.0
        best_ckpt_path = os.path.join(self.output_dir, "best_checkpoint.pt")
        history: List[Dict[str, Any]] = []
        print(f"\n[Execution Engine] Target: {self.device.type.upper()} | Model: {self.config_mode.upper()} | Dataset: {self.dataset_name}")

        for epoch in range(1, epochs + 1):
            train_stats = self._train_epoch(epoch)
            val_metrics = self._evaluate_split(self.val_loader)
            row = {"epoch": epoch, **train_stats, **{f"val_{key}": value for key, value in val_metrics.items()}}
            history.append(row)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_stats['loss']:.4f} | Val Acc: {val_metrics['overall_acc']:.2f}%")
            if val_metrics["overall_acc"] > best_val_acc:
                best_val_acc = val_metrics["overall_acc"]
                torch.save(self.model.state_dict(), best_ckpt_path)

        if os.path.exists(best_ckpt_path):
            self.model.load_state_dict(torch.load(best_ckpt_path, map_location=self.device, weights_only=True))
            self.model.invalidate_candidate_cache()

        test_metrics, predictions = self._evaluate_split_detailed(self.test_loader)
        profile_batch = next(iter(self.test_loader), None)
        if profile_batch is None:
            prof_metrics = {"error": "test split is empty", "gflops": "N/A"}
        else:
            prof_metrics = HardwareDeploymentProfiler.profile_model(
                self.model,
                profile_batch["image"][:1].to(self.device),
                profile_batch["question"][0],
            )
            if self.config_mode == "m5":
                self._generate_visualizations(profile_batch)

        self._export_json_outputs(test_metrics, prof_metrics, predictions, history)
        return {"test_metrics": test_metrics, "profiling": prof_metrics, "history": history}

    def _choose_task(self) -> str:
        """Stochastically select one pretraining objective when requested."""

        if self.objective_schedule == "supervised":
            return "answer"
        choices = ["mlm", "inter"]
        if self.config_mode == "m5":
            choices.append("sgs")
        return self.random.choice(choices)

    def _answer_targets(self, answers: List[str]) -> torch.Tensor:
        normalized = [MedicalAnswerNormalizer.normalize(answer) for answer in answers]
        return torch.tensor([self.ans2idx.get(answer, self.unknown_index) for answer in normalized], device=self.device)

    @staticmethod
    def _grounding_targets(bboxes: torch.Tensor, patch_count: int, device: torch.device) -> torch.Tensor:
        """Rasterize normalized boxes into the patch heatmap delta from Equation (10)."""

        batch_size = bboxes.shape[0]
        grid = int(math.sqrt(patch_count))
        targets = torch.zeros((batch_size, patch_count), dtype=torch.float32, device=device)
        if grid * grid != patch_count:
            return targets
        for row, box in enumerate(bboxes.tolist()):
            x1, y1, x2, y2 = box
            if x1 < 0 or x2 <= x1 or y2 <= y1:
                continue
            left = max(0, min(grid - 1, int(math.floor(x1 * grid))))
            top = max(0, min(grid - 1, int(math.floor(y1 * grid))))
            right = max(left + 1, min(grid, int(math.ceil(x2 * grid))))
            bottom = max(top + 1, min(grid, int(math.ceil(y2 * grid))))
            mask = torch.zeros((grid, grid), device=device)
            mask[top:bottom, left:right] = 1.0
            targets[row] = mask.flatten()
        return targets

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.model.invalidate_candidate_cache()
        totals: Dict[str, float] = {"loss": 0.0, "loss_ans": 0.0, "loss_mlm": 0.0, "loss_inter": 0.0, "loss_sgs": 0.0}

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} Training"):
            images = batch["image"].to(self.device)
            task = self._choose_task()
            need_saliency = self.config_mode == "m5" and (task in {"answer", "sgs"})
            outputs = self.model(
                images,
                list(batch["question"]),
                list(batch["organ"]),
                return_saliency=need_saliency,
                mask_text=(task == "mlm"),
            )
            target_indices = self._answer_targets(list(batch["answer"])) if task == "answer" else None
            grounding = None
            if "grounding_bbox" in batch and outputs.get("saliency_map") is not None:
                grounding = self._grounding_targets(batch["grounding_bbox"], outputs["saliency_map"].shape[-1], self.device)

            self.optimizer.zero_grad(set_to_none=True)
            losses = self.loss_engine(
                outputs,
                target_indices=target_indices,
                saliency_maps=outputs.get("saliency_map"),
                grounding_targets=grounding,
                task=task,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            for key in totals:
                value = losses.get(key, 0.0)
                totals[key] += float(value.detach().item() if isinstance(value, torch.Tensor) else value)

        denominator = max(1, len(self.train_loader))
        return {key: value / denominator for key, value in totals.items()}

    def _evaluate_split(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        targets: List[str] = []
        predictions: List[str] = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(batch["image"].to(self.device), list(batch["question"]), list(batch["organ"]))
                predictions.extend(self._prediction_strings(outputs["logits"]))
                targets.extend(MedicalAnswerNormalizer.normalize(answer) for answer in batch["answer"])
        return {"overall_acc": classification_metrics(targets, predictions)["overall_acc"]}

    def _prediction_strings(self, logits: torch.Tensor) -> List[str]:
        indices = logits.argmax(dim=-1).detach().cpu().tolist()
        unknown = MedicalAnswerNormalizer.normalize("[UNK]")
        return [self.answer_vocab[index] if index < len(self.answer_vocab) else unknown for index in indices]

    def _evaluate_split_detailed(self, loader: DataLoader) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self.model.eval()
        targets: List[str] = []
        predictions: List[str] = []
        answer_types: List[str] = []
        levels: List[str] = []
        logs: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(batch["image"].to(self.device), list(batch["question"]), list(batch["organ"]))
                batch_predictions = self._prediction_strings(outputs["logits"])
                for index, prediction in enumerate(batch_predictions):
                    target = MedicalAnswerNormalizer.normalize(batch["answer"][index])
                    answer_type = str(batch["answer_type"][index]).lower()
                    level = str(batch["level"][index])
                    is_correct = prediction == target and not MedicalAnswerNormalizer.is_unknown(target)
                    targets.append(target)
                    predictions.append(prediction)
                    answer_types.append(answer_type)
                    levels.append(level)
                    logs.append({
                        "question": batch["question"][index],
                        "ground_truth": target,
                        "predicted": prediction,
                        "correct": is_correct,
                        "answer_type": answer_type,
                        "level": level,
                        "organ": batch["organ"][index],
                    })

        metrics = classification_metrics(targets, predictions, answer_types=answer_types, levels=levels)
        if self.dataset_name == "kvasir_vqa_x1":
            metrics.update(generation_metrics(targets, predictions, compute_bertscore=self.compute_bertscore))
        elif self.dataset_name in {"vqa_rad", "slake", "path_vqa"}:
            metrics = {key: metrics.get(key, 0.0) for key in ("open_acc", "close_acc", "overall_acc", "macro_f1")}
        elif self.dataset_name == "kvasir_vqa":
            metrics = {key: metrics.get(key, 0.0) for key in ("overall_acc", "macro_f1", "precision", "recall")}
        return metrics, logs

    def _generate_visualizations(self, batch: Dict[str, Any]) -> None:
        vis_dir = os.path.join(self.output_dir, "visualizations")
        with torch.no_grad():
            outputs = self.model(
                batch["image"].to(self.device),
                list(batch["question"]),
                list(batch["organ"]),
                return_saliency=True,
            )
        if "saliency_map" not in outputs:
            return
        for index in range(min(4, len(batch["image"]))):
            path = os.path.join(vis_dir, f"sample_{index:03d}_contrastive_overlay.png")
            OrganSaliencyVisualizer.save_overlay(
                batch["image"][index],
                outputs["saliency_map"][index],
                path,
                title=f"Q: {batch['question'][index]}\nOrgan: {batch['organ'][index]}",
            )

    def _export_json_outputs(
        self,
        test_metrics: Dict[str, Any],
        prof_metrics: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> None:
        """Write machine-readable versions of Tables I--VI and the sample log."""

        table_i = dataset_characteristics(self.dataset_name)
        table_ii_row = build_table_ii_row(self.dataset_name, self.config_mode, test_metrics)
        table_iii_row = build_table_iii_row(self.dataset_name, self.config_mode, test_metrics)
        table_v_row = {
            "model_config": self.config_mode.upper(),
            "shared_single_tower": self.config_mode != "m1",
            "clinical_anchor_tokens": self.config_mode in {"m3", "m4", "m5"},
            "dynamic_negative_sampling": self.config_mode in {"m4", "m5"},
            "soft_contrastive_saliency": self.config_mode == "m5",
            **test_metrics,
        }
        table_vi_row = {
            "patch_size": getattr(self.model.patch_embed, "patch_size", None),
            "image_resolution": getattr(self.model.patch_embed, "img_size", None),
            "visual_token_count": getattr(self.model.patch_embed, "num_patches", None),
            **test_metrics,
            "latency_ms": prof_metrics.get("latency_ms_mean"),
            "peak_gpu_vram_gb": prof_metrics.get("peak_vram_gb"),
        }

        write_json(os.path.join(self.output_dir, "table_i_dataset_protocol.json"), table_i)
        write_json(os.path.join(self.output_dir, "table_ii_classical.json"), {"references": REFERENCE_TABLE_II, "ours": [table_ii_row]})
        write_json(os.path.join(self.output_dir, "table_iii_gastrointestinal.json"), {"references": REFERENCE_TABLE_III, "ours": [table_iii_row]})
        write_json(os.path.join(self.output_dir, "table_iv_efficiency.json"), prof_metrics)
        write_json(os.path.join(self.output_dir, "table_v_ablation.json"), [table_v_row])
        write_json(os.path.join(self.output_dir, "table_vi_sensitivity.json"), [table_vi_row])
        write_json(os.path.join(self.output_dir, "table_results.json"), test_metrics)
        write_json(os.path.join(self.output_dir, "run_history.json"), history)
        write_json(os.path.join(self.output_dir, "run_config.json"), {
            "dataset": self.dataset_name,
            "model": self.config_mode,
            "answer_vocab_size": len(self.answer_vocab),
            "objective_schedule": self.objective_schedule,
            "seed": self.seed,
        })
        write_json(os.path.join(self.output_dir, "predictions.json"), predictions)
        write_rows_csv(os.path.join(self.output_dir, "table_v_ablation.csv"), [table_v_row])
        write_rows_csv(os.path.join(self.output_dir, "table_vi_sensitivity.csv"), [table_vi_row])
