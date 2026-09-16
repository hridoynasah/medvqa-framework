# MedVQA Framework

Unified training and evaluation for VQA-RAD, SLAKE, PathVQA, Kvasir-VQA, and Kvasir-VQA-x1 with model configurations M1--M5.

## Run one experiment

```bash
uv run python main.py --dataset vqa_rad --model m5 --batch-size 32 --img-size 224 --epochs 20
```

All model and data controls are exposed through the command line:

```bash
uv run python main.py --help
```

For a deterministic CPU-safe run:

```bash
uv run python main.py --dataset vqa_rad --model m5 --smoke-test
```

Run all five configurations and aggregate Table V:

```bash
uv run python main.py --dataset path_vqa --ablation --epochs 20
```

Run Table VI patch/resolution sensitivity settings:

```bash
uv run python main.py --dataset vqa_rad --model m5 --sensitivity-grid 32:224,16:224,16:384,14:384 --epochs 20
```

The run directory contains `table_i_dataset_protocol.json` through `table_vi_sensitivity.json`, CSV versions of the ablation/sensitivity tables, predictions, checkpoints, profiling metrics, and run metadata.

`--bertscore` enables the optional BERTScore calculation for Kvasir-VQA-x1. It may download a language model on first use.

SLAKE stores image filenames in its QA rows and publishes the actual images in `imgs.zip`. The adapter downloads and extracts that archive automatically into the cache; `--image-root` can be used to point to an existing extracted `imgs` directory.

The real-data adapter loads Hugging Face images lazily, so it does not decode every image into RAM during dataset construction. Kvasir-VQA-x1 stores image references separately from its QA annotations. You may supply a local image directory; otherwise referenced images are downloaded on demand into the default user cache or the directory supplied with `--image-cache-dir`.

```bash
uv run python main.py --dataset kvasir_vqa_x1 --image-root ./Kvasir-VQA/images --model m5 --epochs 20
```

To control the on-demand image cache explicitly:

```bash
uv run python main.py --dataset kvasir_vqa_x1 --image-cache-dir ./data/kvasir_vqa_x1_cache --model m5 --epochs 20
```
