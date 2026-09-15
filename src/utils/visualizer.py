import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class OrganSaliencyVisualizer:
    @staticmethod
    def save_overlay(
        image_tensor: torch.Tensor,
        saliency_vector: torch.Tensor,
        output_path: str,
        title: str = ""
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        grid_dim = int(np.sqrt(saliency_vector.shape[-1]))
        sal_grid = saliency_vector.view(grid_dim, grid_dim).detach().cpu().numpy()

        sal_resized = cv2.resize(sal_grid, (img_np.shape[1], img_np.shape[0]))
        denom = (sal_resized.max() - sal_resized.min())
        sal_norm = (sal_resized - sal_resized.min()) / (denom if denom > 1e-7 else 1.0)

        heatmap = cv2.applyColorMap(np.uint8(255 * sal_norm), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        overlay = 0.6 * img_np + 0.4 * heatmap[:, :, ::-1]
        overlay = np.clip(overlay, 0, 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(overlay)
        plt.axis("off")
        if title:
            plt.title(title, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()