"""Deployment measurements used by Table IV and Table VI."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import torch


class HardwareDeploymentProfiler:
    """Measure full-model parameters, FLOPs, latency, and peak GPU memory."""

    @staticmethod
    def profile_model(model: torch.nn.Module, sample_image: torch.Tensor, sample_question: str) -> Dict[str, Any]:
        device = next(model.parameters()).device
        model.eval()
        total_params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / 1e6

        # FLOPs are traced over the complete image-plus-question forward call.
        gflops: Any = "N/A"
        try:
            from fvcore.nn import FlopCountAnalysis

            class FullModelWrapper(torch.nn.Module):
                def __init__(self, wrapped_model: torch.nn.Module):
                    super().__init__()
                    self.wrapped_model = wrapped_model

                def forward(self, image: torch.Tensor):
                    return self.wrapped_model(image, [sample_question])["logits"]

            # Equation: GFLOPs = total floating-point operations / 10^9.
            gflops = round(FlopCountAnalysis(FullModelWrapper(model), sample_image).total() / 1e9, 2)
        except Exception:
            # Unsupported tokenizer/attention operators are reported honestly rather than replaced by a baseline.
            gflops = "N/A"

        latencies: List[float] = []
        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device=device)

        with torch.no_grad():
            for _ in range(3):
                model(sample_image, [sample_question])
            for _ in range(10):
                if is_cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(sample_image, [sample_question])
                    end.record()
                    torch.cuda.synchronize()
                    latencies.append(start.elapsed_time(end))
                else:
                    started = time.perf_counter()
                    model(sample_image, [sample_question])
                    latencies.append((time.perf_counter() - started) * 1000.0)

        # Peak memory is the maximum allocated during warm-up plus measured inference.
        peak_vram_gb = 0.0
        if is_cuda:
            peak_vram_gb = round(torch.cuda.max_memory_allocated(device=device) / (1024**3), 3)

        patch_embed = getattr(model, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        image_size = getattr(patch_embed, "img_size", None)
        visual_tokens = getattr(patch_embed, "num_patches", None)
        mean_latency = round(float(sum(latencies) / max(1, len(latencies))), 2)
        return {
            "total_params_M": round(total_params, 2),
            "trainable_params_M": round(trainable_params, 2),
            "gflops": gflops,
            "inference_flops_g": gflops,
            "latency_ms_mean": mean_latency,
            "latency_ms_per_sample": mean_latency,
            "peak_vram_gb": peak_vram_gb,
            "peak_memory_gb": peak_vram_gb,
            "visual_encoder": "Linear Patch Projection",
            "text_encoder": "WordPiece / BPE",
            "fusion": "Unified Single-Tower Self-Attention",
            "image_resolution": image_size,
            "patch_size": patch_size,
            "visual_token_count": visual_tokens,
        }

