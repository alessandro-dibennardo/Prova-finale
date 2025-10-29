## CascadeNER Inference Overview

The scripts in this directory provide a lightweight two-stage pipeline for generating CascadeNER predictions with your own extractor and classifier checkpoints. The workflow assumes the DynamicNER dataset has already been prepared in `../DynamicNER_process` as described in the project README.

### 1. Stage‑1 Extraction

Run `extract.sh` to produce JSONL responses that contain the entity extraction prompts and model outputs.

```bash
cd CascadeNER
DATASET_PATH=/path/to/dynamic/classify/en/dev.json \
MODEL_PATH=./model/extractor/your_extractor_ckpt \
CUDA_VISIBLE_DEVICES=0 \
bash extract.sh
```

The script stores the generated JSONLs in `./model/extractor/infer_result/` by default. Set `OUTPUT_DIR` to change the destination. All parameters (Swift binary path, model path, dataset path, device, sample size) are overridable via environment variables – see the comments in `extract.sh`.

### 2. Stage‑2 Classification

Aggregate the Stage‑1 outputs and classify each entity with the classifier model:

```bash
python model/infer.py \
  --responses_dir ./model/extractor/infer_result \
  --classifier_model ./model/classifier/your_classifier_ckpt \
  --category_file ../DynamicNER_process/DynamicNER.json \
  --output_file outputs/cascade_predictions.json \
  --device cuda
```

Additional arguments include `--max_new_tokens`, `--temperature`, and `--limit` to control decoding behaviour and the number of processed sentences.

### 3. Evaluation

Compare the predictions against a ground-truth JSON (BASE format) using:

```bash
python evaluate.py \
  path/to/ground_truth.json \
  outputs/cascade_predictions.json
```

This prints micro-averaged precision/recall/F1 both for `(entity, category)` pairs and entity-only metrics.

### 4. Single Query Demo

For quick experiments with a single sentence/entity pair:

```bash
python demo.py \
  --model ./model/classifier/your_classifier_ckpt \
  --category_file ../DynamicNER_process/DynamicNER.json \
  --sentence "Kobe is out." \
  --entity "Kobe"
```

### Model Placement

Place extractor checkpoints under `model/extractor/` and classifier checkpoints under `model/classifier/`. The subfolders `please_put_models_here/` mark the expected locations.

### Notes

- Stage‑1 assumes the extractor is run with ModelScope Swift’s `infer` command; adjust `extract.sh` if using another interface.
- The classification pipeline expects responses formatted with `query`/`response` keys and entities wrapped in `##...##`.
- For large batches, consider running Stage‑2 on GPU (`--device cuda`) with `torch_dtype=float16` automatically enabled.
