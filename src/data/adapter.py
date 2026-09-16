"""Dataset adapters that normalize all supported sources to one sample schema."""

from __future__ import annotations

import hashlib
import io
import os
import random
import urllib.error
import urllib.parse
import urllib.request
import zipfile
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


def _open_image_file(path: Path) -> Image.Image:
    """Open an image and detach it from the file handle before returning it."""

    with Image.open(path) as image:
        return image.convert("RGB")


def _download_remote_image(url: str, cache_dir: Path) -> Image.Image:
    """Download one remote image into a cache and return a detached RGB image."""

    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower() or ".jpg"
    cache_name = hashlib.sha256(url.encode("utf-8")).hexdigest() + suffix
    cache_path = cache_dir / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not cache_path.is_file():
        temporary_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "medvqa-framework/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            temporary_path.write_bytes(payload)
            temporary_path.replace(cache_path)
        except (OSError, urllib.error.URLError) as exc:
            if not cache_path.is_file():
                raise UniversalMedVQADataError(
                    f"Could not download image '{url}' into '{cache_dir}': {exc}"
                ) from exc
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    try:
        return _open_image_file(cache_path)
    except (OSError, ValueError) as exc:
        raise UniversalMedVQADataError(
            f"Downloaded image is not readable: '{cache_path}' (source: {url})"
        ) from exc


def _resolve_image(
    value: Any,
    row: Dict[str, Any],
    image_root: Optional[str],
    image_cache_dir: Optional[str] = None,
) -> Any:
    """Resolve embedded images, local references, or Kvasir-VQA-x1 URLs."""

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        value = value.get("path")
        if value is None:
            return None

    if not isinstance(value, str):
        if value is None:
            return None
        return Image.fromarray(np.asarray(value)).convert("RGB")

    parsed = urllib.parse.urlparse(value)
    is_remote = parsed.scheme in {"http", "https"}
    image_id = str(_first_value(row, ("img_id", "image_id"), "") or "")
    candidates: List[Path] = []
    if image_root:
        root = Path(image_root)
        candidates.extend([root / value, root / Path(parsed.path).name])
        for extension in (".jpg", ".jpeg", ".png", ".bmp"):
            candidates.append(root / f"{image_id}{extension}")
    elif not is_remote:
        candidates.append(Path(value))

    for candidate in candidates:
        if candidate.is_file():
            return _open_image_file(candidate)

    if is_remote:
        default_cache = Path.home() / ".cache" / "medvqa-framework" / "kvasir-vqa-x1"
        return _download_remote_image(value, Path(image_cache_dir) if image_cache_dir else default_cache)
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
        return _sample_to_item(self.samples[idx], self.transform)


def _sample_to_item(item: StandardizedMedVQASample, transform=None) -> Dict[str, Any]:
    """Convert one canonical sample into the stable batch schema."""

    image = transform(item.image) if transform else transforms.ToTensor()(item.image)
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


class LazyHuggingFaceMedVQADataset(Dataset):
    """Dataset view that decodes one Hugging Face image only when it is requested."""

    def __init__(
        self,
        source_dataset: Any,
        indices: Sequence[int],
        transform=None,
        dataset_name: str = "dataset",
        default_answer_type: Optional[str] = None,
        image_root: Optional[str] = None,
        image_cache_dir: Optional[str] = None,
    ):
        self.source_dataset = source_dataset
        self.indices = list(indices)
        self.transform = transform
        self.dataset_name = dataset_name
        self.default_answer_type = default_answer_type
        self.image_root = image_root
        self.image_cache_dir = image_cache_dir

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.source_dataset[self.indices[idx]]
        sample = UniversalMedVQADataModule._row_to_sample(
            row,
            dataset_name=self.dataset_name,
            default_answer_type=self.default_answer_type,
            image_root=self.image_root,
            image_cache_dir=self.image_cache_dir,
        )
        return _sample_to_item(sample, self.transform)


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
        image_cache_dir: Optional[str] = None,
    ) -> List[StandardizedMedVQASample]:
        return [
            cls._row_to_sample(
                row,
                dataset_name=dataset_name,
                default_answer_type=default_answer_type,
                image_root=image_root,
                image_cache_dir=image_cache_dir,
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_sample(
        row: Dict[str, Any],
        dataset_name: str,
        default_answer_type: Optional[str] = None,
        image_root: Optional[str] = None,
        image_cache_dir: Optional[str] = None,
    ) -> StandardizedMedVQASample:
        """Normalize one row; real images are decoded only for this row."""

        image_value = _first_value(
            row,
            ("image", "img", "pixel_values", "img_name", "image_path", "image_file", "filename"),
        )
        image = _resolve_image(image_value, row, image_root, image_cache_dir)
        if image is None:
            image_id = _first_value(row, ("img_id", "image_id"), "")
            raise UniversalMedVQADataError(
                f"{dataset_name}: could not resolve image for row {image_id!r}. "
                "For image-reference datasets, provide a valid --image-root or allow URL caching."
            )
        question = _first_value(row, ("question", "Question", "query", "text"))
        answer = _first_value(row, ("answer", "Answer", "answers", "label", "response"), "")
        if question is None:
            raise UniversalMedVQADataError(f"{dataset_name}: row has no question field")
        answer_type = _first_value(row, ("answer_type", "question_type", "type"), default_answer_type)
        level = _first_value(row, ("level", "question_level", "reasoning_level", "complexity", "diagnostic_level"), "")
        organ = _first_value(row, ("organ", "anatomy", "body_part"))
        bbox = _first_value(row, ("bbox", "bounding_box", "box", "grounding_bbox", "region"))
        return StandardizedMedVQASample(image, question, answer, answer_type, organ, level, bbox)

    @staticmethod
    def _prepare_dataset_image_root(
        dataset_name: str,
        image_root: Optional[str],
        image_cache_dir: Optional[str],
    ) -> Optional[str]:
        """Prepare image assets for datasets whose Hub rows contain file names only."""

        if dataset_name != "slake" or image_root:
            return image_root

        cache_root = (
            Path(image_cache_dir)
            if image_cache_dir
            else Path.home() / ".cache" / "medvqa-framework" / "slake"
        )
        extracted_root = cache_root / "extracted"
        image_root_path = extracted_root / "imgs"
        marker = extracted_root / ".imgs_complete"
        if marker.is_file() and image_root_path.is_dir():
            return str(image_root_path)

        try:
            from huggingface_hub import hf_hub_download

            archive_path = hf_hub_download(
                repo_id="BoKelvin/SLAKE",
                filename="imgs.zip",
                repo_type="dataset",
                cache_dir=str(cache_root / "huggingface"),
            )
            extracted_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extracted_root)
            if not image_root_path.is_dir():
                raise UniversalMedVQADataError(
                    f"SLAKE image archive did not contain the expected directory '{image_root_path}'."
                )
            marker.write_text("complete", encoding="utf-8")
            return str(image_root_path)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            raise UniversalMedVQADataError(
                "SLAKE image assets could not be downloaded or extracted. "
                "You can provide an extracted image directory with --image-root."
            ) from exc

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

    @staticmethod
    def _grouped_train_val_test_indices(source_dataset: Any, seed: int) -> Tuple[List[int], List[int], List[int]]:
        """Split row indices by image ID without decoding the image column."""

        metadata_columns = [column for column in source_dataset.column_names if column != "image"]
        metadata = source_dataset.select_columns(metadata_columns)
        group_column = next(
            (column for column in ("img_id", "image_id") if column in metadata.column_names),
            None,
        )
        if group_column is None:
            group_values = [str(index) for index in range(len(metadata))]
        else:
            group_values = [str(value) for value in metadata[group_column]]

        groups: Dict[str, List[int]] = {}
        for index, group_id in enumerate(group_values):
            groups.setdefault(group_id, []).append(index)
        group_items = list(groups.values())
        random.Random(seed).shuffle(group_items)
        test_count = max(1, int(len(group_items) * 0.15)) if len(group_items) > 2 else 0
        val_count = max(1, int(len(group_items) * 0.15)) if len(group_items) > 2 else 0
        test_groups = group_items[:test_count]
        val_groups = group_items[test_count : test_count + val_count]
        train_groups = group_items[test_count + val_count :]
        flatten = lambda selected: [index for group in selected for index in group]
        return flatten(train_groups), flatten(val_groups), flatten(test_groups)

    @staticmethod
    def _metadata_view(source_dataset: Any) -> Any:
        """Return a Hugging Face view with the image column removed."""

        columns = [column for column in source_dataset.column_names if column != "image"]
        return source_dataset.select_columns(columns)

    @staticmethod
    def _filtered_indices_by_language(metadata: Any, indices: Sequence[int]) -> List[int]:
        """Keep English SLAKE rows without decoding its image column."""

        if "q_lang" not in metadata.column_names:
            return list(indices)
        languages = metadata["q_lang"]
        return [
            index
            for index in indices
            if str(languages[index] or "en").lower() in {"en", "english"}
        ]

    @staticmethod
    def _answer_vocabulary(metadata: Any, indices: Sequence[int], max_answer_candidates: int) -> List[str]:
        """Build the answer vocabulary from metadata only, never from decoded images."""

        answer_column = next(
            (column for column in ("answer", "Answer", "answers", "label", "response") if column in metadata.column_names),
            None,
        )
        counts: Counter[str] = Counter()
        if answer_column is not None:
            answers = metadata[answer_column]
            for index in indices:
                answer = MedicalAnswerNormalizer.normalize(_answer_text(answers[index]))
                if answer:
                    counts[answer] += 1
        ordered = sorted(counts, key=lambda answer: (-counts[answer], answer))
        if max_answer_candidates > 0:
            ordered = ordered[: max(1, max_answer_candidates - 1)]
        return ["[UNK]"] + ordered

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
        image_cache_dir: Optional[str] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
        """Return lazy train/validation/test views and a deterministic answer vocabulary."""

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
            source = raw["raw"]
            metadata = cls._metadata_view(source)
            train_indices, validation_indices, test_indices = cls._grouped_train_val_test_indices(source, seed)
            train_source = validation_source = test_source = source
            train_metadata = validation_metadata = test_metadata = metadata
        else:
            if "train" not in raw or "test" not in raw:
                raise UniversalMedVQADataError(f"Dataset '{hub_id}' must provide train and test splits")
            train_source = raw["train"]
            test_source = raw["test"]
            validation_source = raw.get("validation")
            train_metadata = cls._metadata_view(train_source)
            test_metadata = cls._metadata_view(test_source)
            validation_metadata = cls._metadata_view(validation_source) if validation_source is not None else None
            train_indices = list(range(len(train_source)))
            validation_indices = list(range(len(validation_source))) if validation_source is not None else []
            test_indices = list(range(len(test_source)))

        if dataset_name == "vqa_rad":
            if max_samples > 0:
                train_indices = train_indices[:max_samples]
            rng = random.Random(seed)
            rng.shuffle(train_indices)
            val_count = max(1, int(len(train_indices) * 0.15)) if len(train_indices) > 1 else 0
            validation_indices, train_indices = train_indices[:val_count], train_indices[val_count:]
        elif dataset_name == "slake":
            train_indices = cls._filtered_indices_by_language(train_metadata, train_indices)
            validation_indices = cls._filtered_indices_by_language(validation_metadata, validation_indices)
            test_indices = cls._filtered_indices_by_language(test_metadata, test_indices)

        if max_samples > 0 and dataset_name != "vqa_rad":
            train_indices = train_indices[:max_samples]

        if not validation_indices:
            rng = random.Random(seed)
            shuffled = list(train_indices)
            rng.shuffle(shuffled)
            val_count = max(1, int(len(shuffled) * 0.15)) if len(shuffled) > 1 else 0
            validation_indices, train_indices = shuffled[:val_count], shuffled[val_count:]
            validation_source = train_source
            validation_metadata = train_metadata

        if not train_indices or not validation_indices or not test_indices:
            raise UniversalMedVQADataError(
                f"Invalid split sizes for {dataset_name}: train={len(train_indices)}, "
                f"val={len(validation_indices)}, test={len(test_indices)}"
            )

        answer_vocab = cls._answer_vocabulary(train_metadata, train_indices, max_answer_candidates)
        default_answer_type = "open" if dataset_name == "kvasir_vqa_x1" else None
        resolved_image_root = cls._prepare_dataset_image_root(dataset_name, image_root, image_cache_dir)
        return (
            LazyHuggingFaceMedVQADataset(
                train_source,
                train_indices,
                transform,
                dataset_name,
                default_answer_type,
                resolved_image_root,
                image_cache_dir,
            ),
            LazyHuggingFaceMedVQADataset(
                validation_source,
                validation_indices,
                transform,
                dataset_name,
                default_answer_type,
                resolved_image_root,
                image_cache_dir,
            ),
            LazyHuggingFaceMedVQADataset(
                test_source,
                test_indices,
                transform,
                dataset_name,
                default_answer_type,
                resolved_image_root,
                image_cache_dir,
            ),
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
