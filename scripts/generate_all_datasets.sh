#!/bin/bash
# Script to generate all dataset variants used in experiments
# This allows TAs/reviewers to reproduce all datasets

set -e

echo "=========================================="
echo "Generating All Dataset Variants"
echo "=========================================="
echo ""

# Check if source data exists
if [ ! -f "data/sample_triples.jsonl" ]; then
    echo "ERROR: data/sample_triples.jsonl not found!"
    echo "Please ensure the repository is complete."
    exit 1
fi

# Check how many samples are available
TOTAL_SAMPLES=$(wc -l < data/sample_triples.jsonl)
echo "Available samples in data/sample_triples.jsonl: $TOTAL_SAMPLES"
echo ""

# If not enough samples for paper_eval (needs 200), generate synthetic data
if [ "$TOTAL_SAMPLES" -lt 200 ]; then
    echo "Not enough samples for paper_eval dataset. Generating synthetic data..."
    python3 scripts/generate_synthetic_data.py \
      --output data/synthetic_triples.jsonl \
      --num_triples 200 \
      --seed 42
    echo "Using synthetic data for paper_eval dataset."
    TRIPLES_FILE="data/synthetic_triples.jsonl"
else
    TRIPLES_FILE="data/sample_triples.jsonl"
fi

if [ ! -f "data/entity_texts.jsonl" ]; then
    echo "WARNING: data/entity_texts.jsonl not found!"
    echo "SimCSE variants will not work without this file."
fi

# 1. Basic hybrid dataset
echo "1. Generating basic hybrid dataset (data/hybrid/)..."
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 50 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
echo "  ✓ Created data/hybrid/"
echo ""

# 2. Hybrid large dataset
echo "2. Generating large hybrid dataset (data/hybrid_large/)..."
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid_large \
  --limit 100 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
echo "  ✓ Created data/hybrid_large/"
echo ""

    # 3. Hybrid with SimCSE (if entity_texts.jsonl exists)
if [ -f "data/entity_texts.jsonl" ]; then
    echo "3. Generating hybrid dataset with SimCSE (data/hybrid_simcse/)..."
    python3 scripts/prepare_hybrid_dataset.py \
      --triples_jsonl data/sample_triples.jsonl \
      --out_dir data/hybrid_simcse \
      --limit 50 \
      --use_sns \
      --entity_texts_jsonl data/entity_texts.jsonl \
      --sns_top_k 5 \
      --sns_threshold 0.8 \
      --train_ratio 0.8 \
      --val_ratio 0.1 \
      --test_ratio 0.1 \
      --seed 42
    echo "  ✓ Created data/hybrid_simcse/"
    echo ""

    # 4. Hybrid SimCSE with default settings
    echo "4. Generating hybrid SimCSE default dataset (data/hybrid_simcse_default/)..."
    python3 scripts/prepare_hybrid_dataset.py \
      --triples_jsonl data/sample_triples.jsonl \
      --out_dir data/hybrid_simcse_default \
      --limit 50 \
      --use_sns \
      --entity_texts_jsonl data/entity_texts.jsonl \
      --train_ratio 0.8 \
      --val_ratio 0.1 \
      --test_ratio 0.1 \
      --seed 42
    echo "  ✓ Created data/hybrid_simcse_default/"
    echo ""
else
    echo "3-4. Skipping SimCSE variants (entity_texts.jsonl not found)"
    echo ""
fi

# 5. Paper evaluation dataset
echo "5. Generating paper evaluation dataset (data/paper_eval/)..."
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl "$TRIPLES_FILE" \
  --out_dir data/paper_eval \
  --limit 200 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --no_images
echo "  ✓ Created data/paper_eval/"
TRAIN_COUNT=$(wc -l < data/paper_eval/train.jsonl)
VAL_COUNT=$(wc -l < data/paper_eval/val.jsonl)
TEST_COUNT=$(wc -l < data/paper_eval/test.jsonl)
echo "    Train: $TRAIN_COUNT, Val: $VAL_COUNT, Test: $TEST_COUNT samples"
echo ""

echo "=========================================="
echo "✓ All datasets generated successfully!"
echo "=========================================="
echo ""
echo "Generated datasets:"
echo "  - data/hybrid/ (basic, 50 samples)"
echo "  - data/hybrid_large/ (100 samples)"
if [ -f "data/entity_texts.jsonl" ]; then
    echo "  - data/hybrid_simcse/ (with SimCSE, threshold 0.8)"
    echo "  - data/hybrid_simcse_default/ (with SimCSE, default settings)"
fi
echo "  - data/paper_eval/ (200 samples, final evaluation dataset)"
echo ""
echo "To verify, check line counts:"
echo "  wc -l data/*/train.jsonl data/*/val.jsonl data/*/test.jsonl"

