import pytest
import torch
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

if __name__ == "__main__":
    test_imports_and_instantiation()
    test_non_degenerate_candidate_embeddings()
    test_organ_parsing_and_saliency_no_crash()
    print("All unit verification tests passed successfully!")
