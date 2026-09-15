"""Evaluation helpers for classification and free-form medical VQA."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MedicalAnswerNormalizer:
    """Normalize answers while keeping the operation deterministic and inspectable."""

    ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    PUNCTUATION = re.compile(r"[^\w\s./+-]", re.UNICODE)
    WHITESPACE = re.compile(r"\s+", re.UNICODE)

    @classmethod
    def normalize(cls, answer: Any) -> str:
        """Return the canonical answer string used by vocabulary and metrics."""

        text = str(answer).lower().strip()
        text = cls.ARTICLES.sub(" ", text)
        text = cls.PUNCTUATION.sub(" ", text)
        text = cls.WHITESPACE.sub(" ", text).strip()
        return text

    @classmethod
    def is_unknown(cls, answer: Any) -> bool:
        """Use the normalized spelling consistently across training and reporting."""

        return cls.normalize(answer) in {"unk", "unknown", "<unk>"}


def safe_accuracy(targets: Sequence[str], predictions: Sequence[str]) -> float:
    """Compute percentage accuracy, returning zero for an empty split."""

    if not targets:
        return 0.0
    return float(accuracy_score(targets, predictions) * 100.0)


def classification_metrics(
    targets: Sequence[str],
    predictions: Sequence[str],
    answer_types: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute the metrics required by Tables II, III, and V."""

    targets = list(targets)
    predictions = list(predictions)
    result: Dict[str, Any] = {
        "overall_acc": round(safe_accuracy(targets, predictions), 2),
        "macro_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    if targets:
        result["macro_f1"] = round(float(f1_score(targets, predictions, average="macro", zero_division=0)), 4)
        result["precision"] = round(float(precision_score(targets, predictions, average="macro", zero_division=0)), 4)
        result["recall"] = round(float(recall_score(targets, predictions, average="macro", zero_division=0)), 4)

    if answer_types is not None:
        answer_types = [str(value).lower() for value in answer_types]
        for label, output_key in (("open", "open_acc"), ("closed", "close_acc")):
            indices = [index for index, value in enumerate(answer_types) if value == label]
            result[output_key] = round(
                safe_accuracy([targets[index] for index in indices], [predictions[index] for index in indices]),
                2,
            )

    if levels is not None:
        for level in sorted({str(value) for value in levels}):
            indices = [index for index, value in enumerate(levels) if str(value) == level]
            result[f"level_{level}_acc"] = round(
                safe_accuracy([targets[index] for index in indices], [predictions[index] for index in indices]),
                2,
            )

    return result


def generation_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    compute_bertscore: bool = False,
) -> Dict[str, Optional[float]]:
    """Compute BLEU-4, ROUGE-L, METEOR, and optional BERTScore.

    BERTScore requires a model download and is therefore opt-in.  The other
    three metrics use local packages already declared by this project.
    """

    references = [MedicalAnswerNormalizer.normalize(value) for value in references]
    hypotheses = [MedicalAnswerNormalizer.normalize(value) for value in hypotheses]
    if not references:
        return {"bleu4": 0.0, "rouge_l": 0.0, "meteor": 0.0, "bertscore": None}

    bleu4 = 0.0
    meteor = 0.0
    rouge_l = 0.0

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoother = SmoothingFunction().method1
        scores = [
            sentence_bleu([reference.split()], hypothesis.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
            for reference, hypothesis in zip(references, hypotheses)
        ]
        bleu4 = float(np.mean(scores)) if scores else 0.0
    except Exception:
        bleu4 = 0.0

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(reference, hypothesis)["rougeL"].fmeasure for reference, hypothesis in zip(references, hypotheses)]
        rouge_l = float(np.mean(scores)) if scores else 0.0
    except Exception:
        rouge_l = 0.0

    try:
        from nltk.translate.meteor_score import meteor_score

        scores = [meteor_score([reference.split()], hypothesis.split()) for reference, hypothesis in zip(references, hypotheses)]
        meteor = float(np.mean(scores)) if scores else 0.0
    except Exception:
        meteor = 0.0

    bertscore: Optional[float] = None
    if compute_bertscore:
        try:
            import evaluate

            metric = evaluate.load("bertscore")
            output = metric.compute(predictions=hypotheses, references=references, lang="en", verbose=False)
            bertscore = float(np.mean(output["f1"]))
        except Exception:
            # The report remains valid when the optional BERTScore model is offline.
            bertscore = None

    return {
        "bleu4": round(bleu4, 4),
        "rouge_l": round(rouge_l, 4),
        "meteor": round(meteor, 4),
        "bertscore": None if bertscore is None else round(bertscore, 4),
    }

