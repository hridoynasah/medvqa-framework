"""Answer, MLM, inter-image, and soft-grounding objectives."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedVQALossEngine(nn.Module):
    """Implement Equations (11), (13), (16), and (17) from the specification."""

    def __init__(self, config_mode: str = "m5", lambda_sgs: float = 0.1, lambda_inter: float = 0.01):
        super().__init__()
        self.config_mode = config_mode.lower()
        self.lambda_sgs = lambda_sgs
        self.lambda_inter = lambda_inter
        self.answer_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)

    @staticmethod
    def _soft_grounding_loss(saliency_maps: torch.Tensor, grounding_targets: Optional[torch.Tensor]) -> torch.Tensor:
        """Equation (11): -log(sum_p delta_p * M_contrast[p] + epsilon)."""

        if grounding_targets is None or saliency_maps.numel() == 0:
            return saliency_maps.new_zeros(())
        targets = grounding_targets.to(device=saliency_maps.device, dtype=saliency_maps.dtype)
        valid = targets.sum(dim=-1) > 0
        if not bool(valid.any()):
            return saliency_maps.new_zeros(())
        targets = targets / targets.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        overlap = (targets * saliency_maps).sum(dim=-1)
        return (-torch.log(overlap.clamp(min=1e-7))[valid]).mean()

    @staticmethod
    def _dynamic_inter_loss(
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Equations (15)--(17): batch similarity, dynamic K, and hard negatives."""

        batch_size = visual_features.shape[0]
        if batch_size <= 1:
            zero = visual_features.new_zeros(())
            return zero, zero, zero, 0

        visual = F.normalize(visual_features, dim=-1)
        text = F.normalize(text_features, dim=-1)
        similarity = visual @ text.t()
        positive = similarity.diag()

        # gamma_align is the mean positive alignment; gamma_uniform is log E[e^S].
        gamma_align = positive.mean()
        gamma_uniform = torch.logsumexp(similarity.reshape(-1), dim=0) - torch.log(
            torch.tensor(float(similarity.numel()), device=similarity.device)
        )

        negative_count = batch_size - 1
        # Equation (17): dynamically choose the hard-negative pool size K.
        raw_k = negative_count * torch.cos((torch.pi / 8.0) * (2.0 + gamma_uniform - gamma_align))
        k = int(torch.round(raw_k.detach()).clamp(min=1, max=negative_count).item())

        masked = similarity.clone()
        masked.fill_diagonal_(-torch.inf)
        hard_values, hard_indices = torch.topk(masked, k=k, dim=1)
        del hard_values
        selected = torch.full_like(similarity, -torch.inf)
        selected.scatter_(1, hard_indices, similarity.gather(1, hard_indices))
        selected.diagonal().copy_(similarity.diag())
        targets = torch.arange(batch_size, device=similarity.device)

        # The symmetric direction prevents one modality from dominating the alignment.
        reverse = selected.t()
        loss = 0.5 * (F.cross_entropy(selected, targets) + F.cross_entropy(reverse, targets))
        return loss, gamma_align.detach(), gamma_uniform.detach(), k

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        target_indices: Optional[torch.Tensor] = None,
        saliency_maps: Optional[torch.Tensor] = None,
        grounding_targets: Optional[torch.Tensor] = None,
        task: str = "answer",
    ) -> Dict[str, torch.Tensor | int | str]:
        """Compute one selected objective plus the configured M4/M5 auxiliary terms."""

        task = task.lower()
        logits = model_outputs["logits"]
        zero = logits.new_zeros(())
        loss_answer = zero
        loss_mlm = zero
        loss_inter = zero
        loss_sgs = zero
        gamma_align = zero
        gamma_uniform = zero
        hard_negative_k = 0

        if task in {"answer", "all"}:
            if target_indices is None:
                raise ValueError("target_indices are required for the answer objective")
            # Equation (13): cross entropy over the candidate answer space.
            loss_answer = self.answer_loss(logits, target_indices)
        elif task == "mlm":
            labels = model_outputs["mlm_labels"].reshape(-1)
            if bool((labels != -100).any().detach()):
                loss_mlm = self.mlm_loss(
                    model_outputs["mlm_logits"].reshape(-1, model_outputs["mlm_logits"].shape[-1]),
                    labels,
                )
        elif task == "inter":
            loss_inter, gamma_align, gamma_uniform, hard_negative_k = self._dynamic_inter_loss(
                model_outputs["inter_visual"], model_outputs["inter_text"]
            )
        elif task == "sgs":
            if saliency_maps is not None:
                loss_sgs = self._soft_grounding_loss(saliency_maps, grounding_targets)
        else:
            raise ValueError(f"Unknown objective task: {task}")

        if self.config_mode in {"m4", "m5"} and task in {"answer", "all"}:
            loss_inter, gamma_align, gamma_uniform, hard_negative_k = self._dynamic_inter_loss(
                model_outputs["inter_visual"], model_outputs["inter_text"]
            )
        if self.config_mode == "m5" and task in {"answer", "all"} and saliency_maps is not None:
            loss_sgs = self._soft_grounding_loss(saliency_maps, grounding_targets)

        total = loss_answer + loss_mlm + self.lambda_inter * loss_inter + self.lambda_sgs * loss_sgs
        return {
            "loss": total,
            "loss_ans": loss_answer.detach(),
            "loss_mlm": loss_mlm.detach(),
            "loss_inter": loss_inter.detach(),
            "loss_sgs": loss_sgs.detach(),
            "gamma_align": gamma_align,
            "gamma_uniform": gamma_uniform,
            "hard_negative_k": hard_negative_k,
            "task": task,
        }
