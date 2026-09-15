import torch
import torch.nn as nn

class LinearPatchEmbedding(nn.Module):
    """Linear patch projection implementing Equations (1)--(2)."""
    def __init__(self, img_size: int = 384, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        if img_size <= 0 or patch_size <= 0 or img_size < patch_size:
            raise ValueError("img_size and patch_size must be positive, with img_size >= patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        # Equation (1): non-overlapping patch count on the usable image grid.
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            raise ValueError(
                f"Expected images of shape ({self.img_size}, {self.img_size}), got {tuple(x.shape[-2:])}"
            )
        patches = self.proj(x).flatten(2).transpose(1, 2)
        if patches.shape[1] != self.num_patches:
            raise RuntimeError(f"Patch projection produced {patches.shape[1]} patches; expected {self.num_patches}")
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = x + self.pos_embed + self.type_embed
        return x
