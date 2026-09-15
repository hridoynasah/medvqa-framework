"""Unified single-tower models M1--M5."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..data.organ_lexicon import OrganLexiconParser
from .patch_embed import LinearPatchEmbedding


class SingleTowerTransformerLayer(nn.Module):
    """Pre-norm-free Transformer block matching the equations in the specification."""

    def __init__(self, d_model: int = 768, nhead: int = 12, dim_feedforward: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Equation (5)--(6): Q, K, V attention and a row-normalized attention matrix.
        attended, attention = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        src = self.norm1(src + self.dropout1(attended))
        feed_forward = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        # Equation (7): residual FFN update followed by LayerNorm.
        src = self.norm2(src + self.dropout2(feed_forward))
        return src, attention


class SingleTowerMedVQA(nn.Module):
    """M1--M5 implementation with one shared representation contract."""

    def __init__(
        self,
        config_mode: str = "m5",
        img_size: int = 384,
        patch_size: int = 16,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        max_text_len: int = 48,
        vocab_size: int = 30522,
        answer_candidates: Optional[List[str]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config_mode = config_mode.lower()
        if self.config_mode not in {"m1", "m2", "m3", "m4", "m5"}:
            raise ValueError("config_mode must be one of m1, m2, m3, m4, m5")
        if d_model <= 0 or nhead <= 0 or d_model % nhead != 0:
            raise ValueError("d_model must be positive and divisible by nhead")
        self.d_model = d_model
        self.max_text_len = max_text_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.answer_candidates = [str(answer).lower().strip() for answer in (answer_candidates or ["[UNK]", "yes", "no"])]

        # The tokenizer supplies WordPiece IDs; the framework trains the shared embeddings end-to-end.
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.patch_embed = LinearPatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=d_model)

        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_text_len, d_model))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.visual_sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.text_eos_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.text_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.visual_sep_token, std=0.02)
        nn.init.trunc_normal_(self.text_eos_token, std=0.02)

        # M3--M5: learnable type signal added at detected anatomical subwords.
        self.anchor_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.anchor_type_embed, std=0.02)

        self.answer_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.layers = nn.ModuleList([
            SingleTowerTransformerLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout)
            for _ in range(num_layers)
        ])

        if self.config_mode == "m1":
            self.m1_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        # Equation (12): tau is parameterized in log space to keep it positive.
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))
        self.cached_candidate_embeddings: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        """Clear trainable candidate representations whenever mode changes."""

        self.invalidate_candidate_cache()
        return super().train(mode)

    def _apply(self, fn):
        # ``Module.to(device)`` also invalidates a cache created on another device.
        self.invalidate_candidate_cache()
        return super()._apply(fn)

    def invalidate_candidate_cache(self) -> None:
        self.cached_candidate_embeddings = None

    def _mask_text_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 15% conditional MLM masking while preserving labels elsewhere as -100."""

        labels = torch.full_like(input_ids, -100)
        eligible = attention_mask.bool()
        probability = torch.full(input_ids.shape, 0.15, device=input_ids.device)
        mask_positions = torch.bernoulli(probability).bool() & eligible
        # Keep at least one masked token for a non-empty question.
        for row in range(input_ids.shape[0]):
            candidates = torch.where(eligible[row])[0]
            if len(candidates) and not mask_positions[row].any():
                mask_positions[row, candidates[0]] = True
        labels[mask_positions] = input_ids[mask_positions]

        masked = input_ids.clone()
        mask_id = self.tokenizer.mask_token_id or 103
        replace_with_mask = mask_positions & (torch.rand_like(probability) < 0.8)
        random_tokens = torch.randint(0, self.vocab_size, input_ids.shape, device=input_ids.device)
        replace_random = mask_positions & ~replace_with_mask & (torch.rand_like(probability) < 0.5)
        masked[replace_with_mask] = mask_id
        masked[replace_random] = random_tokens[replace_random]
        return masked, labels

    def encode_text(
        self,
        text_list: List[str],
        device: torch.device,
        is_query: bool = True,
        mask_text: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor]:
        """Implement Equation (3) and optionally generate C-MLM labels."""

        encoding = self.tokenizer(
            text_list,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(device)
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        labels = torch.full_like(input_ids, -100)
        if mask_text:
            input_ids, labels = self._mask_text_ids(input_ids, attention_mask)

        tokens = self.text_embed(input_ids) + self.text_pos_embed + self.text_type_embed
        if is_query and self.config_mode in {"m3", "m4", "m5"}:
            for batch_index, row in enumerate(encoding.input_ids.tolist()):
                anchor_indices = OrganLexiconParser.get_organ_token_indices(row, self.tokenizer)
                for token_index in anchor_indices:
                    if token_index < self.max_text_len:
                        tokens[batch_index, token_index] = tokens[batch_index, token_index] + self.anchor_type_embed.squeeze(0)
        return tokens, attention_mask, encoding.input_ids.tolist(), labels

    def get_candidate_embeddings(self, device: torch.device) -> torch.Tensor:
        """Encode each answer phrase and apply masked mean pooling for p_c."""

        if self.cached_candidate_embeddings is not None:
            return self.cached_candidate_embeddings

        answer_tokens, answer_mask, _, _ = self.encode_text(self.answer_candidates, device, is_query=False)
        expanded_mask = answer_mask.unsqueeze(-1).float()
        # The masked mean keeps candidate meaning instead of using a shared CLS row.
        pooled = (answer_tokens * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1.0)
        projected = self.answer_proj(pooled)
        normalized = F.normalize(projected, dim=-1)
        if not self.training:
            self.cached_candidate_embeddings = normalized.detach()
        return normalized

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        images: torch.Tensor,
        questions: List[str],
        organs: Optional[List[str]] = None,
        return_saliency: bool = False,
        mask_text: bool = False,
    ) -> Dict[str, Any]:
        del organs  # Organ positions are recovered from the encoded query for consistent behavior.
        batch_size = images.shape[0]
        device = images.device

        visual_tokens = self.patch_embed(images)
        text_tokens, text_mask, raw_input_ids, mlm_labels = self.encode_text(
            questions, device, is_query=True, mask_text=mask_text
        )
        visual_pre_features = visual_tokens[:, 1:].mean(dim=1)
        text_pre_features = self._masked_mean(text_tokens, text_mask)
        last_attention = None

        if self.config_mode == "m1":
            # M1: dual stream; the same Transformer weights process each modality separately.
            for layer in self.layers:
                visual_tokens, _ = layer(visual_tokens)
                text_tokens, _ = layer(text_tokens, key_padding_mask=(text_mask == 0))
            z_cls = self.m1_fusion(torch.cat([visual_tokens[:, 0], text_tokens[:, 0] if text_tokens.shape[1] else visual_tokens[:, 0]], dim=-1))
            text_hidden = text_tokens
        else:
            # Equation (4): [visual CLS, patches, SEP, text tokens, EOS].
            visual_sep = self.visual_sep_token.expand(batch_size, -1, -1)
            text_eos = self.text_eos_token.expand(batch_size, -1, -1)
            z = torch.cat([visual_tokens, visual_sep, text_tokens, text_eos], dim=1)
            visual_mask = torch.ones((batch_size, visual_tokens.shape[1] + 1), dtype=torch.bool, device=device)
            eos_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            joint_mask = torch.cat([visual_mask, text_mask.bool(), eos_mask], dim=1)
            for layer_index, layer in enumerate(self.layers):
                z, attention = layer(z, key_padding_mask=~joint_mask)
                if layer_index == self.num_layers - 1:
                    last_attention = attention
            z_cls = z[:, 0]
            text_start = visual_tokens.shape[1] + 1
            text_hidden = z[:, text_start : text_start + self.max_text_len]
            visual_tokens = z[:, : visual_tokens.shape[1]]

        candidate_embeddings = self.get_candidate_embeddings(device)
        temperature = self.log_tau.exp().clamp(min=0.01, max=1.0)
        logits = torch.matmul(F.normalize(z_cls, dim=-1), candidate_embeddings.t()) / temperature

        output: Dict[str, Any] = {
            "logits": logits,
            "z_cls": z_cls,
            "v_tokens": visual_tokens,
            "t_tokens": text_hidden,
            "inter_visual": F.normalize(visual_pre_features, dim=-1),
            "inter_text": F.normalize(text_pre_features, dim=-1),
            "mlm_logits": self.mlm_head(text_hidden),
            "mlm_labels": mlm_labels,
            "text_mask": text_mask,
        }

        if self.config_mode == "m5" and return_saliency and last_attention is not None:
            output["saliency_map"] = self._compute_contrastive_saliency(
                last_attention,
                raw_input_ids,
                visual_tokens.shape[1] - 1,
                device,
            )
        return output

    def _compute_contrastive_saliency(
        self,
        last_attention: torch.Tensor,
        raw_input_ids: List[List[int]],
        patch_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Equation (9): ReLU(organ-to-patch attention minus null-to-patch attention)."""

        batch_size = last_attention.shape[0]
        saliency_maps = torch.zeros((batch_size, patch_count), device=device)
        text_start = patch_count + 2  # visual CLS + patches + visual SEP
        eos_index = text_start + self.max_text_len

        for batch_index in range(batch_size):
            organ_index = OrganLexiconParser.get_organ_token_index(raw_input_ids[batch_index], self.tokenizer)
            if organ_index is not None:
                query_index = text_start + organ_index
                alpha_organ = last_attention[batch_index, query_index, 1 : patch_count + 1]
                alpha_null = last_attention[batch_index, eos_index, 1 : patch_count + 1]
                contrastive = F.relu(alpha_organ - alpha_null)
                source = contrastive if bool((contrastive.sum() > 0).detach()) else alpha_organ
            else:
                source = last_attention[batch_index, 0, 1 : patch_count + 1]
            # Normalize the patch distribution so Equation (11) has a true probability mass.
            saliency_maps[batch_index] = source / source.sum().clamp(min=1e-7)
        return saliency_maps


def build_medvqa_model(
    config_mode: str,
    answer_candidates: List[str],
    img_size: int = 384,
    patch_size: int = 16,
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 12,
    max_text_len: int = 48,
    smoke_test: bool = False,
) -> SingleTowerMedVQA:
    """Build a requested configuration, scaling only model width/depth in smoke mode."""

    if smoke_test:
        d_model, nhead, num_layers = 128, 4, 2
    return SingleTowerMedVQA(
        config_mode=config_mode,
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_text_len=max_text_len,
        answer_candidates=answer_candidates,
    )
