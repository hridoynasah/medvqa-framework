"""Small deterministic clinical lexicon used by the anchor-token module."""

from __future__ import annotations

from typing import Iterable, List, Optional, Set


class OrganLexiconParser:
    """Extract anatomical entities while separating them from pathologies."""

    ANATOMICAL_LEXICON: Set[str] = {
        "colon", "polyp", "polyps", "esophagus", "stomach", "cecum", "rectum", "ileum", "mucosa", "lumen",
        "lung", "lungs", "heart", "liver", "kidney", "kidneys", "brain", "chest", "abdomen", "head", "neck",
        "spine", "mediastinum", "pleural", "diaphragm", "ventricle", "aorta", "trachea", "rib", "ribs", "bone", "pelvis",
        "cell", "cells", "nucleus", "nuclei", "tissue", "gland", "stroma", "epithelium", "lymph", "node",
    }
    PATHOLOGY_LEXICON: Set[str] = {
        "esophagitis", "adenoma", "ulcerative", "colitis", "carcinoma", "lesion", "tumor",
    }
    ORGAN_LEXICON: Set[str] = ANATOMICAL_LEXICON | PATHOLOGY_LEXICON

    @classmethod
    def parse_organ(cls, text: str) -> Optional[str]:
        """Prefer an anatomical match, then fall back to a pathology entity."""

        tokens = str(text).lower().replace("?", " ").replace(".", " ").replace(",", " ").split()
        for token in tokens:
            if token in cls.ANATOMICAL_LEXICON:
                return token
        for token in tokens:
            if token in cls.PATHOLOGY_LEXICON:
                return token
        return None

    @classmethod
    def _decoded_token(cls, token_id: int, tokenizer) -> str:
        token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        token = str(token).lower().replace("##", "")
        return token.strip(".,?!:;()[]{}")

    @classmethod
    def get_organ_token_indices(cls, input_ids: Iterable[int], tokenizer) -> List[int]:
        """Return all encoded content-token positions matching the lexicon."""

        indices: List[int] = []
        for index, token_id in enumerate(input_ids):
            token = cls._decoded_token(int(token_id), tokenizer)
            if token in cls.ORGAN_LEXICON:
                indices.append(index)
        return indices

    @classmethod
    def get_organ_token_index(cls, input_ids: Iterable[int], tokenizer) -> Optional[int]:
        """Return the first anatomical token, or the first pathology token."""

        ids = list(input_ids)
        indices = cls.get_organ_token_indices(ids, tokenizer)
        if not indices:
            return None
        anatomical = []
        pathology = []
        for index in indices:
            token = cls._decoded_token(int(ids[index]), tokenizer)
            (anatomical if token in cls.ANATOMICAL_LEXICON else pathology).append(index)
        return (anatomical or pathology)[0]

    @classmethod
    def extract_anchor_indices(cls, token_ids: List[int], tokenizer) -> List[int]:
        """Backward-compatible alias used by earlier experiments."""

        return cls.get_organ_token_indices(token_ids, tokenizer)
