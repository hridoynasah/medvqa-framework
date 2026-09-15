"""Command-line entry point for one run, an ablation, or a sensitivity grid."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.adapter import UniversalMedVQADataModule
from src.engine.trainer import SingleRunExecutionEngine
from src.models.single_tower import build_medvqa_model
from src.utils.reporting import write_json, write_rows_csv


def seed_everything(seed: int) -> None:
    """Seed every local RNG used by the data pipeline and model initialization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loaders(train_set, val_set, test_set, batch_size: int, seed: int, num_workers: int, device: str | None):
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device is not None and device.startswith("cuda") or device is None and torch.cuda.is_available()
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )


def parse_args() -> argparse.Namespace:
    """Expose all experiment controls through the run command."""

    parser = argparse.ArgumentParser(description="Unified Med-VQA M1--M5 experiment runner")
    parser.add_argument("--dataset", required=True, choices=["vqa_rad", "slake", "path_vqa", "kvasir_vqa", "kvasir_vqa_x1"])
    parser.add_argument("--model", default="m5", choices=["m1", "m2", "m3", "m4", "m5"])
    parser.add_argument("--ablation", action="store_true", help="Run M1 through M5 and aggregate Table V")
    parser.add_argument("--sensitivity-grid", default=None, help="Comma-separated PATCH:IMAGE settings for Table VI")
    parser.add_argument("--smoke-test", action="store_true", help="Use deterministic mock data and a small model")
    parser.add_argument("--dataset-id", default=None, help="Override the Hugging Face dataset identifier")
    parser.add_argument("--image-root", default=None, help="Local image directory for datasets storing image IDs/references")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--max-text-len", type=int, default=48)
    parser.add_argument("--max-answer-candidates", type=int, default=0, help="0 means all training answers")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit training rows for controlled experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--objective-schedule", choices=["supervised", "stochastic"], default="supervised")
    parser.add_argument("--bertscore", action="store_true", help="Compute optional BERTScore for Kvasir-VQA-x1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or a torch device string")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_sensitivity_grid(value: str | None, default_patch: int, default_image: int) -> List[Tuple[int, int]]:
    """Parse ``patch:image`` pairs while retaining a normal single-run default."""

    if not value:
        return [(default_patch, default_image)]
    settings: List[Tuple[int, int]] = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid sensitivity setting '{item}'. Expected PATCH:IMAGE")
        patch, image = int(parts[0]), int(parts[1])
        if patch <= 0 or image <= 0:
            raise ValueError(f"Sensitivity values must be positive: '{item}'")
        settings.append((patch, image))
    return settings


def _run_setting(
    args: argparse.Namespace,
    models: Sequence[str],
    image_size: int,
    patch_size: int,
    output_root: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run all requested model configurations for one Table VI setting."""

    train_set, val_set, test_set, answer_vocab = UniversalMedVQADataModule.load_dataset(
        dataset_name=args.dataset,
        smoke_test=args.smoke_test,
        image_size=image_size,
        seed=args.seed,
        dataset_id=args.dataset_id,
        max_answer_candidates=args.max_answer_candidates,
        max_samples=args.max_samples,
        image_root=args.image_root,
    )
    completed: Dict[str, Any] = {}
    aggregate_rows: List[Dict[str, Any]] = []
    batch_size = 4 if args.smoke_test else args.batch_size
    epochs = 1 if args.smoke_test else args.epochs

    for model_name in models:
        seed_everything(args.seed)
        train_loader, val_loader, test_loader = _make_loaders(
            train_set, val_set, test_set, batch_size, args.seed, args.num_workers, args.device
        )
        model = build_medvqa_model(
            config_mode=model_name,
            answer_candidates=answer_vocab,
            img_size=image_size,
            patch_size=patch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_text_len=args.max_text_len,
            smoke_test=args.smoke_test,
        )
        model_output = os.path.join(output_root, model_name) if args.ablation else output_root
        engine = SingleRunExecutionEngine(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            answer_vocab=answer_vocab,
            output_dir=model_output,
            config_mode=model_name,
            dataset_name=args.dataset,
            device=args.device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            objective_schedule=args.objective_schedule,
            compute_bertscore=args.bertscore,
            seed=args.seed,
        )
        result = engine.execute_single_run(epochs=epochs)
        completed[model_name] = result
        metrics = result["test_metrics"]
        aggregate_rows.append({
            "model_config": model_name.upper(),
            "shared_single_tower": model_name != "m1",
            "clinical_anchor_tokens": model_name in {"m3", "m4", "m5"},
            "dynamic_negative_sampling": model_name in {"m4", "m5"},
            "soft_contrastive_saliency": model_name == "m5",
            "patch_size": patch_size,
            "image_resolution": image_size,
            "visual_token_count": result["profiling"].get("visual_token_count"),
            "latency_ms": result["profiling"].get("latency_ms_mean"),
            "peak_memory_gb": result["profiling"].get("peak_memory_gb"),
            **metrics,
        })
        print(f"[Completed {model_name.upper()} @ {patch_size}:{image_size}] {metrics}")
    return completed, aggregate_rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    default_image = 128 if args.smoke_test and args.img_size == 384 else args.img_size
    settings = _parse_sensitivity_grid(args.sensitivity_grid, args.patch_size, default_image)
    models = ["m1", "m2", "m3", "m4", "m5"] if args.ablation else [args.model]
    root_output = args.output_dir or os.path.join("outputs", args.dataset, "ablation" if args.ablation else args.model)
    all_rows: List[Dict[str, Any]] = []
    all_results: Dict[str, Any] = {}

    print("=" * 60)
    print(f"Dataset: {args.dataset} | Models: {', '.join(models)}")
    print(f"Settings: {settings} | Batch: {4 if args.smoke_test else args.batch_size} | Epochs: {1 if args.smoke_test else args.epochs}")
    print(f"Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')} | Smoke: {args.smoke_test}")
    print("=" * 60)

    for patch_size, image_size in settings:
        setting_output = root_output if len(settings) == 1 else os.path.join(root_output, f"patch{patch_size}_img{image_size}")
        results, rows = _run_setting(args, models, image_size, patch_size, setting_output)
        all_results[f"patch{patch_size}_img{image_size}"] = {name: result["test_metrics"] for name, result in results.items()}
        all_rows.extend(rows)

    if args.ablation:
        write_json(os.path.join(root_output, "table_v_ablation.json"), all_rows)
        write_rows_csv(os.path.join(root_output, "table_v_ablation.csv"), all_rows)

    # Table VI contains one row per requested patch/resolution setting. In an ablation,
    # the selected --model is preferred when available; otherwise the M5 row is used.
    sensitivity_rows = []
    for row in all_rows:
        if row["model_config"].lower() == args.model:
            sensitivity_rows.append(row)
    if not sensitivity_rows:
        sensitivity_rows = [row for row in all_rows if row["model_config"] == "M5"] or all_rows[:1]
    write_json(os.path.join(root_output, "table_vi_sensitivity.json"), sensitivity_rows)
    write_rows_csv(os.path.join(root_output, "table_vi_sensitivity.csv"), sensitivity_rows)
    write_json(os.path.join(root_output, "experiment_summary.json"), {
        "dataset": args.dataset,
        "models": models,
        "settings": [{"patch_size": patch, "image_size": image} for patch, image in settings],
        "seed": args.seed,
        "results": all_results,
    })


if __name__ == "__main__":
    main()
