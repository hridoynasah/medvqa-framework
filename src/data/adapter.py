"""Dataset adapters that normalize all supported sources to one sample schema."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.metrics import MedicalAnswerNormalizer
from .organ_lexicon import OrganLexiconParser


DATASET_HUB_IDS = {
    "vqa_rad": "flaviagiammarino/vqa-rad",
    "slake": "BoKelvin/SLAKE",
    "path_vqa": "flaviagiammarino/path-vqa",
    # These identifiers can be overridden with --dataset-id when using a private mirror.
    "kvasir_vqa": "SimulaMet-HOST/Kvasir-VQA",
    "kvasir_vqa_x1": "SimulaMet/Kvasir-VQA-x1",
}


def _first_value(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Read the first available field so minor Hub-schema differences stay isolated."""

    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _answer_text(value: Any) -> str:
    """Convert scalar, list, or dictionary answer fields into one answer string."""

    if isinstance(value, dict):
        value = _first_value(value, ("answer", "text", "value"), "")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def _resolve_image(value: Any, row: Dict[str, Any], image_root: Optional[str]) -> Any:
    """Resolve local image references used by Kvasir-VQA-x1."""

    if not isinstance(value, str):
        return value
    candidates = [Path(value)]
    if image_root:
        root = Path(image_root)
        candidates.append(root / value)
        image_id = _first_value(row, ("img_id", "image_id"), "")
        for extension in (".jpg", ".jpeg", ".png", ".bmp"):
            candidates.append(root / f"{image_id}{extension}")
    for candidate in candidates:
        if candidate.is_file():
            with Image.open(candidate) as image:
                return image.convert("RGB")
    return None


def _normalise_bbox(value: Any, image: Image.Image) -> Tuple[float, float, float, float]:
    """Convert common box formats to normalized ``x1,y1,x2,y2`` coordinates."""

    if value is None:
        return (-1.0, -1.0, -1.0, -1.0)
    if isinstance(value, dict):
        if all(key in value for key in ("x", "y", "width", "height")):
            value = [value["x"], value["y"], value["x"] + value["width"], value["y"] + value["height"]]
        else:
            value = _first_value(value, ("bbox", "box", "coordinates"), None)
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return (-1.0, -1.0, -1.0, -1.0)
    x1, y1, x2, y2 = [float(item) for item in value[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0:
        width, height = image.size
        x1, x2 = x1 / max(width, 1), x2 / max(width, 1)
        y1, y2 = y1 / max(height, 1), y2 / max(height, 1)
    return tuple(float(max(0.0, min(1.0, item))) for item in (x1, y1, x2, y2))


class StandardizedMedVQASample:
    """Canonical representation consumed by every model configuration."""

    def __init__(
        self,
        image: Any,
        question: str,
        answer: Any,
        answer_type: Optional[str] = None,
        organ: Optional[str] = None,
        level: Optional[str] = None,
        grounding_bbox: Any = None,
    ):
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif image is not None:
            pil_image = Image.fromarray(np.asarray(image)).convert("RGB")
        else:
            pil_image = Image.new("RGB", (384, 384), (0, 0, 0))

        self.image = pil_image
        self.question = str(question)
        self.answer = MedicalAnswerNormalizer.normalize(_answer_text(answer))
        if answer_type is None:
            answer_type = "closed" if self.answer in {"yes", "no", "true", "false"} else "open"
        self.answer_type = str(answer_type).lower()
        self.organ = organ or OrganLexiconParser.parse_organ(self.question) or "tissue"
        self.level = str(level) if level is not None else ""
        self.grounding_bbox = _normalise_bbox(grounding_bbox, self.image)


class UniversalMedVQADataset(Dataset):
    """PyTorch dataset with a stable batch schema across all five benchmarks."""

    def __init__(self, samples: List[StandardizedMedVQASample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image = self.transform(item.image) if self.transform else transforms.ToTensor()(item.image)
        return {
            "image": image,
            "question": item.question,
            "answer": item.answer,
            "answer_type": item.answer_type,
            "organ": item.organ,
            "level": item.level,
            # A negative box means that no region annotation was supplied.
            "grounding_bbox": torch.tensor(item.grounding_bbox, dtype=torch.float32),
        }


class UniversalMedVQADataError(ValueError):
    """Raised when a dataset cannot be normalized without guessing."""


class UniversalMedVQADataModule:
    """Load VQA-RAD, SLAKE, PathVQA, Kvasir-VQA, and Kvasir-VQA-x1."""

    @staticmethod
    def get_transforms(image_size: int = 384):
        """Use a shared deterministic transform; laterality-changing flips are avoided."""

        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def _rows_to_samples(
        cls,
        rows: Iterable[Dict[str, Any]],
        dataset_name: str,
        default_answer_type: Optional[str] = None,
        image_root: Optional[str] = None,
    ) -> List[StandardizedMedVQASample]:
        samples: List[StandardizedMedVQASample] = []
        for row in rows:
            image = _resolve_image(_first_value(row, ("image", "img", "pixel_values")), row, image_root)
            question = _first_value(row, ("question", "Question", "query", "text"))
            answer = _first_value(row, ("answer", "Answer", "answers", "label", "response"), "")
            if question is None:
                raise UniversalMedVQADataError(f"{dataset_name}: row has no question field")
            answer_type = _first_value(row, ("answer_type", "question_type", "type"), default_answer_type)
            level = _first_value(row, ("level", "question_level", "reasoning_level", "complexity", "diagnostic_level"), "")
            organ = _first_value(row, ("organ", "anatomy", "body_part"))
            bbox = _first_value(row, ("bbox", "bounding_box", "box", "grounding_bbox", "region"))
            samples.append(StandardizedMedVQASample(image, question, answer, answer_type, organ, level, bbox))
        return samples

    @staticmethod
    def _grouped_train_val_test(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split grouped QA rows so questions from one image cannot cross a split."""

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            group_id = str(_first_value(row, ("img_id", "image_id", "image"), len(groups)))
            groups.setdefault(group_id, []).append(row)
        group_items = list(groups.values())
        random.Random(seed).shuffle(group_items)
        test_count = max(1, int(len(group_items) * 0.15)) if len(group_items) > 2 else 0
        val_count = max(1, int(len(group_items) * 0.15)) if len(group_items) > 2 else 0
        test_groups = group_items[:test_count]
        val_groups = group_items[test_count : test_count + val_count]
        train_groups = group_items[test_count + val_count :]
        flatten = lambda selected: [row for group in selected for row in group]
        return flatten(train_groups), flatten(val_groups), flatten(test_groups)

    @classmethod
    def load_dataset(
        cls,
        dataset_name: str,
        smoke_test: bool = False,
        image_size: int = 384,
        seed: int = 42,
        dataset_id: Optional[str] = None,
        max_answer_candidates: int = 0,
        max_samples: int = 0,
        image_root: Optional[str] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
        """Return train/validation/test datasets and a deterministic answer vocabulary."""

        dataset_name = dataset_name.lower().replace("-", "_")
        if dataset_name not in DATASET_HUB_IDS:
            raise UniversalMedVQADataError(f"Unsupported dataset '{dataset_name}'. Supported: {sorted(DATASET_HUB_IDS)}")
        transform = cls.get_transforms(image_size)

        if smoke_test:
            return cls._build_deterministic_mock_splits(transform, seed, dataset_name, image_size)

        from datasets import load_dataset

        hub_id = dataset_id or DATASET_HUB_IDS[dataset_name]
        raw = load_dataset(hub_id)
        if dataset_name == "kvasir_vqa" and "raw" in raw and "train" not in raw:
            train_rows, validation_rows, test_rows = cls._grouped_train_val_test(list(raw["raw"]), seed)
        else:
            if "train" not in raw or "test" not in raw:
                raise UniversalMedVQADataError(f"Dataset '{hub_id}' must provide train and test splits")
            train_rows = list(raw["train"])
            validation_rows = list(raw.get("validation", []))
            test_rows = list(raw["test"])
        if dataset_name == "kvasir_vqa_x1":
            image_values = [_first_value(row, ("image", "img", "pixel_values")) for row in train_rows[:10]]
            if any(isinstance(value, str) for value in image_values) and not image_root:
                raise UniversalMedVQADataError(
                    "Kvasir-VQA-x1 provides image references. Supply --image-root pointing to the downloaded Kvasir images."
                )
        if max_samples > 0:
            train_rows = train_rows[:max_samples]

        if dataset_name == "vqa_rad":
            rng = random.Random(seed)
            rng.shuffle(train_rows)
            val_count = max(1, int(len(train_rows) * 0.15)) if len(train_rows) > 1 else 0
            validation_rows, train_rows = train_rows[:val_count], train_rows[val_count:]
            train_samples = cls._rows_to_samples(train_rows, dataset_name, image_root=image_root)
            val_samples = cls._rows_to_samples(validation_rows, dataset_name, image_root=image_root)
            test_samples = cls._rows_to_samples(test_rows, dataset_name, image_root=image_root)
        elif dataset_name == "slake":
            filter_english = lambda row: str(row.get("q_lang", "en")).lower() in {"en", "english"}
            train_samples = cls._rows_to_samples([row for row in train_rows if filter_english(row)], dataset_name, image_root=image_root)
            val_samples = cls._rows_to_samples([row for row in validation_rows if filter_english(row)], dataset_name, image_root=image_root)
            test_samples = cls._rows_to_samples([row for row in test_rows if filter_english(row)], dataset_name, image_root=image_root)
        elif dataset_name == "path_vqa":
            train_samples = cls._rows_to_samples(train_rows, dataset_name, image_root=image_root)
            val_samples = cls._rows_to_samples(validation_rows, dataset_name, image_root=image_root)
            test_samples = cls._rows_to_samples(test_rows, dataset_name, image_root=image_root)
        elif dataset_name == "kvasir_vqa_x1":
            train_samples = cls._rows_to_samples(train_rows, dataset_name, default_answer_type="open", image_root=image_root)
            val_samples = cls._rows_to_samples(validation_rows, dataset_name, default_answer_type="open", image_root=image_root)
            test_samples = cls._rows_to_samples(test_rows, dataset_name, default_answer_type="open", image_root=image_root)
        else:
            train_samples = cls._rows_to_samples(train_rows, dataset_name, image_root=image_root)
            val_samples = cls._rows_to_samples(validation_rows, dataset_name, image_root=image_root)
            test_samples = cls._rows_to_samples(test_rows, dataset_name, image_root=image_root)

        if not val_samples:
            rng = random.Random(seed)
            shuffled = list(train_samples)
            rng.shuffle(shuffled)
            val_count = max(1, int(len(shuffled) * 0.15)) if len(shuffled) > 1 else 0
            val_samples, train_samples = shuffled[:val_count], shuffled[val_count:]
        if not train_samples or not val_samples or not test_samples:
            raise UniversalMedVQADataError(
                f"Invalid split sizes for {dataset_name}: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
            )

        counts = Counter(sample.answer for sample in train_samples if sample.answer)
        answers = sorted(counts, key=lambda answer: (-counts[answer], answer))
        if max_answer_candidates > 0:
            answers = answers[: max(1, max_answer_candidates - 1)]
        answer_vocab = ["[UNK]"] + answers
        return (
            UniversalMedVQADataset(train_samples, transform),
            UniversalMedVQADataset(val_samples, transform),
            UniversalMedVQADataset(test_samples, transform),
            answer_vocab,
        )

    @classmethod
    def _build_deterministic_mock_splits(cls, transform, seed: int, dataset_name: str = "vqa_rad", image_size: int = 128):
        """Create repeatable micro-splits used by tests and the CPU smoke command."""

        rng = np.random.RandomState(seed)
        organs = ["colon", "lung", "polyp", "liver", "heart"]

        def make_sample(index: int, answer_type: Optional[str] = None) -> StandardizedMedVQASample:
            image = Image.fromarray(rng.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8))
            organ = organs[index % len(organs)]
            closed = index % 2 == 0
            question = f"Is there any abnormality detected in the {organ}?" if closed else f"What pathology is observed in the {organ}?"
            answer = "yes" if closed else f"{organ}ic lesion"
            if dataset_name == "kvasir_vqa_x1":
                answer_type = "open"
            return StandardizedMedVQASample(
                image,
                question,
                answer,
                answer_type or ("closed" if closed else "open"),
                organ,
                str(index % 3 + 1),
                (0.25, 0.25, 0.75, 0.75),
            )

        train_samples = [make_sample(index) for index in range(16)]
        val_samples = [make_sample(index + 100) for index in range(8)]
        test_samples = [make_sample(index + 200) for index in range(8)]
        answer_vocab = ["[UNK]"] + sorted({sample.answer for sample in train_samples})
        return (
            UniversalMedVQADataset(train_samples, transform),
            UniversalMedVQADataset(val_samples, transform),
            UniversalMedVQADataset(test_samples, transform),
            answer_vocab,
        )
