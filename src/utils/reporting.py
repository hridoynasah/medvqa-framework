"""Dataset metadata and report-table construction."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List


DATASET_CHARACTERISTICS: Dict[str, Dict[str, Any]] = {
    "vqa_rad": {"dataset": "VQA-RAD", "imaging_modality": "CT, MRI, X-ray", "anatomical_scope": "Head, Chest, Abdomen", "images": 315, "total_qa_pairs": 3515, "question_categories": "Open-ended (637), Closed-ended (878)", "primary_metrics": "Open Acc, Closed Acc, Overall Acc (%)"},
    "slake": {"dataset": "SLAKE", "imaging_modality": "CT, MRI, X-ray", "anatomical_scope": "39 organs, 12 diseases", "images": 642, "total_qa_pairs": 14028, "question_categories": "Knowledge-based, Vision-only", "primary_metrics": "Open Acc, Closed Acc, Overall Acc (%)"},
    "path_vqa": {"dataset": "PathVQA", "imaging_modality": "Histopathology, Microscopy", "anatomical_scope": "Diverse biopsy tissues", "images": 4998, "total_qa_pairs": 32799, "question_categories": "Open-ended (16,465), Closed-ended (16,334)", "primary_metrics": "Open Acc, Closed Acc, Overall Acc (%)"},
    "kvasir_vqa": {"dataset": "Kvasir-VQA", "imaging_modality": "GI video endoscopy", "anatomical_scope": "Esophagus, Stomach, Colon, Rectum", "images": 6500, "total_qa_pairs": 58849, "question_categories": "Diagnostic class, Multi-choice, Counting", "primary_metrics": "Overall Acc, Macro-F1, Precision, Recall"},
    "kvasir_vqa_x1": {"dataset": "Kvasir-VQA-x1", "imaging_modality": "Endoscopy (normal and corrupted)", "anatomical_scope": "Upper and lower GI tract", "images": 6500, "total_qa_pairs": 159484, "question_categories": "Stratified clinical reasoning (Levels 1, 2, 3)", "primary_metrics": "BLEU-4, ROUGE-L, METEOR, BERTScore"},
}


REFERENCE_TABLE_II: List[Dict[str, Any]] = [
    {"model": "Nguyen et al. (2019)", "vqa_rad": "38.20 / 69.70 / 53.95"},
    {"model": "Do et al. (2021)", "vqa_rad": "53.70 / 75.80 / 67.00", "path_vqa": "13.40 / 84.00 / 48.80"},
    {"model": "Gong et al. (2022)", "vqa_rad": "56.60 / 79.60 / 70.40", "path_vqa": "13.40 / 83.50 / 48.60"},
    {"model": "Liu et al. (2021)", "vqa_rad": "61.10 / 80.40 / 72.70", "slake": "81.20 / 83.40 / 82.10"},
    {"model": "Pan et al. (2022)", "vqa_rad": "63.80 / 80.30 / 73.30", "path_vqa": "18.20 / 84.40 / 50.40"},
    {"model": "Chen et al. (2022)", "vqa_rad": "67.23 / 83.46 / 77.01", "slake": "80.31 / 87.82 / 83.25"},
    {"model": "Gu et al. (2024)", "vqa_rad": "68.72 / 86.40 / 79.38", "slake": "68.72 / 86.40 / 79.38"},
    {"model": "Li et al. (2023)", "vqa_rad": "71.50 / 84.20 / 79.20", "path_vqa": "39.00 / 90.40 / 65.10"},
    {"model": "Ha et al. (2024)", "vqa_rad": "57.50 / 83.50 / 73.20", "slake": "84.50 / 92.10 / 91.00"},
    {"model": "Xu et al. (2025)", "vqa_rad": "76.00 / 87.90 / 83.10", "slake": "90.50 / 91.80 / 87.50", "path_vqa": "41.40 / 91.50 / 66.50"},
    {"model": "Mbarek et al. (2026)", "vqa_rad": "83.25 / 85.06 / 83.91", "slake": "79.85 / 85.21 / 82.65", "path_vqa": "41.39 / 91.57 / 93.18"},
]


REFERENCE_TABLE_III: List[Dict[str, Any]] = [
    {"benchmark": "Kvasir-VQA", "model": "Eslami et al. (2021)", "accuracy": 74.20, "macro_f1": 71.35, "level_3_acc": None, "bleu4": 0.082, "rouge_l": 0.764, "meteor": 0.385, "bertscore": 0.812},
    {"benchmark": "Kvasir-VQA", "model": "Xiao et al. (2023)", "accuracy": 82.40, "macro_f1": 79.80, "level_3_acc": None, "bleu4": 0.150, "rouge_l": 0.800, "meteor": 0.440, "bertscore": 0.880},
    {"benchmark": "Kvasir-VQA-x1", "model": "Khan et al. (2025)", "accuracy": 0.0401, "macro_f1": 0.0039, "level_3_acc": 0.0004, "bleu4": 0.0045, "rouge_l": 0.0856, "meteor": 0.0622, "bertscore": 0.8351, "note": "Reported BLEU-1 for levels 1-3."},
    {"benchmark": "Kvasir-VQA-x1", "model": "Bai et al. (2024)", "note": "Track 1 values supplied in the source table."},
    {"benchmark": "Kvasir-VQA-x1", "model": "Gautam et al. (2025)", "note": "Track 1 values supplied in the source table."},
]


def dataset_characteristics(dataset_name: str) -> Dict[str, Any]:
    """Return Table I metadata plus the canonical dataset key."""

    key = dataset_name.lower().replace("-", "_")
    return {"dataset_key": key, **DATASET_CHARACTERISTICS.get(key, {"dataset": key})}


def build_table_ii_row(dataset_name: str, model: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Build the current-framework row for the classical benchmark table."""

    return {
        "model": "Ours",
        "architecture": "Unified Single-Tower Multi-Modal Transformer",
        "config": model,
        "dataset": dataset_name.lower().replace("-", "_"),
        "open_acc": metrics.get("open_acc"),
        "close_acc": metrics.get("close_acc"),
        "overall_acc": metrics.get("overall_acc"),
    }


def build_table_iii_row(dataset_name: str, model: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Build the current-framework row for the gastrointestinal benchmark table."""

    return {
        "benchmark": dataset_name,
        "model": "Ours",
        "config": model,
        "accuracy": metrics.get("overall_acc"),
        "macro_f1": metrics.get("macro_f1"),
        "level_1_acc": metrics.get("level_1_acc"),
        "level_2_acc": metrics.get("level_2_acc"),
        "level_3_acc": metrics.get("level_3_acc"),
        "bleu4": metrics.get("bleu4"),
        "rouge_l": metrics.get("rouge_l"),
        "meteor": metrics.get("meteor"),
        "bertscore": metrics.get("bertscore"),
    }


def write_json(path: str, payload: Any) -> None:
    """Write UTF-8 JSON while creating the destination directory."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_rows_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write a list of dictionaries as a flat CSV report."""

    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
