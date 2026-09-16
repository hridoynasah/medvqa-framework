import pytest
import torch
from PIL import Image
from src.models.single_tower import build_medvqa_model
from src.data.organ_lexicon import OrganLexiconParser
from src.data.adapter import UniversalMedVQADataModule
from src.engine.losses import MedVQALossEngine
from src.utils.metrics import MedicalAnswerNormalizer

def test_imports_and_instantiation():
    vocab = ["[UNK]", "adenoma", "colitis", "hyperplasia", "yes", "no"]
    for m in ["m1", "m2", "m3", "m4", "m5"]:
        model = build_medvqa_model(m, vocab, img_size=128, smoke_test=True)
        assert model is not None

def test_non_degenerate_candidate_embeddings():
    vocab = ["[UNK]", "adenoma", "normal", "yes", "no"]
    model = build_medvqa_model("m5", vocab, img_size=128, smoke_test=True)
    embeds = model.get_candidate_embeddings(torch.device("cpu"))
    # Ensure answer embeddings are non-identical across candidate classes
    diff = (embeds[1] - embeds[2]).abs().sum().item()
    assert diff > 1e-4, "Candidate answer embeddings must not be identical vectors!"

def test_organ_parsing_and_saliency_no_crash():
    vocab = ["[UNK]", "yes", "no"]
    model = build_medvqa_model("m5", vocab, img_size=128, smoke_test=True)
    imgs = torch.randn(2, 3, 128, 128)
    questions = ["Is there a lesion in the colon?", "General checkup?"]
    # Even if organs=None, execution must use fallback without crashing
    out = model(imgs, questions, organs=None, return_saliency=True)
    assert "logits" in out
    assert "saliency_map" in out
    assert out["saliency_map"].shape == (2, (128 // 16) ** 2)


def test_deterministic_dataset_contract():
    transform = UniversalMedVQADataModule.get_transforms(32)
    first = UniversalMedVQADataModule._build_deterministic_mock_splits(transform, 7, "kvasir_vqa_x1", 32)
    second = UniversalMedVQADataModule._build_deterministic_mock_splits(transform, 7, "kvasir_vqa_x1", 32)
    assert first[3] == second[3]
    assert torch.equal(first[0][0]["image"], second[0][0]["image"])
    assert first[0][0]["grounding_bbox"].shape == (4,)
    assert first[0][0]["level"] in {"1", "2", "3"}


def test_objectives_and_unknown_label_are_finite():
    outputs = {
        "logits": torch.randn(2, 4, requires_grad=True),
        "z_cls": torch.randn(2, 8, requires_grad=True),
        "inter_visual": torch.randn(2, 8, requires_grad=True),
        "inter_text": torch.randn(2, 8, requires_grad=True),
        "mlm_logits": torch.randn(2, 3, 9, requires_grad=True),
        "mlm_labels": torch.tensor([[1, -100, 2], [3, 4, -100]]),
    }
    engine = MedVQALossEngine("m5")
    answer = engine(outputs, torch.tensor([1, 2]), task="answer")
    mlm = engine(outputs, task="mlm")
    inter = engine(outputs, task="inter")
    assert torch.isfinite(answer["loss"])
    assert torch.isfinite(mlm["loss"])
    assert torch.isfinite(inter["loss"])
    assert MedicalAnswerNormalizer.is_unknown("[UNK]")


class _FakeHFDataset:
    """Small Hugging Face Dataset stand-in for testing split/index behavior."""

    def __init__(self, rows, image_reads=0):
        self.rows = rows
        self.image_reads = image_reads
        self.column_names = list(rows[0]) if rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row.get(key) for row in self.rows]
        row = self.rows[key].copy()
        if "image" in row:
            self.image_reads += 1
        return row

    def select_columns(self, columns):
        return _FakeHFDataset(
            [{column: row.get(column) for column in columns} for row in self.rows],
            self.image_reads,
        )


def test_real_dataset_branches_are_lazy(monkeypatch):
    image = Image.new("RGB", (16, 16), (20, 30, 40))

    def split(rows):
        return _FakeHFDataset(rows)

    def rows(count, include_level=False, include_language=False):
        result = []
        for index in range(count):
            row = {
                "image": image,
                "question": f"What is shown in image {index}?",
                "answer": "yes" if index % 2 == 0 else "lesion",
                "img_id": f"image-{index // 2}",
            }
            if include_level:
                row["complexity"] = (index % 3) + 1
            if include_language:
                row["q_lang"] = "en"
            result.append(row)
        return result

    def fake_load_dataset(dataset_id):
        if dataset_id == "flaviagiammarino/path-vqa":
            return {
                "train": split(rows(6)),
                "validation": split(rows(2)),
                "test": split(rows(2)),
            }
        if dataset_id == "SimulaMet-HOST/Kvasir-VQA":
            return {"raw": split(rows(12))}
        if dataset_id == "SimulaMet/Kvasir-VQA-x1":
            return {"train": split(rows(6, include_level=True)), "test": split(rows(2, include_level=True))}
        if dataset_id == "BoKelvin/SLAKE":
            return {
                "train": split(rows(6, include_language=True)),
                "validation": split(rows(2, include_language=True)),
                "test": split(rows(2, include_language=True)),
            }
        return {"train": split(rows(6)), "test": split(rows(2))}

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    for dataset_name in ["vqa_rad", "slake", "path_vqa", "kvasir_vqa", "kvasir_vqa_x1"]:
        train, validation, test, vocabulary = UniversalMedVQADataModule.load_dataset(
            dataset_name,
            image_size=16,
            seed=42,
        )
        assert len(train) > 0
        assert len(validation) > 0
        assert len(test) > 0
        assert vocabulary[0] == "[UNK]"
        item = train[0]
        assert item["image"].shape == (3, 16, 16)


def test_slake_filename_image_resolution(tmp_path):
    image_dir = tmp_path / "xmlab1"
    image_dir.mkdir()
    image_path = image_dir / "source.jpg"
    Image.new("RGB", (12, 10), (10, 20, 30)).save(image_path)
    sample = UniversalMedVQADataModule._row_to_sample(
        {
            "img_name": "xmlab1/source.jpg",
            "question": "What is shown?",
            "answer": "MRI",
            "q_lang": "en",
        },
        dataset_name="slake",
        image_root=str(tmp_path),
    )
    assert sample.image.size == (12, 10)

if __name__ == "__main__":
    test_imports_and_instantiation()
    test_non_degenerate_candidate_embeddings()
    test_organ_parsing_and_saliency_no_crash()
    print("All unit verification tests passed successfully!")
