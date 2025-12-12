# Hybrid-KG-LLM-Project

Hybrid multi-hop reasoning over knowledge graphs with LLM alignment using Direct Preference Optimization (DPO). This repository provides a complete pipeline for training and evaluating LLMs on knowledge graph reasoning tasks.

## Quick Start: Step-by-Step Guide

Follow these steps in order to reproduce the results:

### Step 1: Clone the Repository

```bash
git clone https://github.com/rlk12345/hybrid-kg-llm-submission
cd hybrid-kg-llm-submission
```

### Step 2: Run Quick Start (Setup)

This sets up the Python environment and installs dependencies:

```bash
bash QUICK_START.sh
```

**What this does:**
- Creates a Python virtual environment
- Installs all required packages
- Verifies the setup is correct

**Time:** 2-5 minutes

### Step 3: Generate All Datasets

This creates all the dataset variants used in experiments:

```bash
bash scripts/generate_all_datasets.sh
```

**What this does:**
- Creates 5 dataset variants in `data/` directories
- Each dataset has train/val/test splits ready for training
- Automatically generates synthetic data if needed (e.g., for paper_eval which needs 200 samples)

**Time:** 1-2 minutes

**Note:** If you see warnings about small datasets, the script will automatically generate synthetic data to ensure you have enough samples for meaningful evaluation.

**Important:** If you previously generated datasets and they have too few samples (e.g., only 2 test samples), you need to regenerate them:
```bash
# Remove old datasets
rm -rf data/paper_eval data/hybrid data/hybrid_large data/hybrid_simcse data/hybrid_simcse_default

# Regenerate with proper sample counts
bash scripts/generate_all_datasets.sh
```

### Step 4: Test the Complete Pipeline (Recommended)

Run a complete end-to-end test to verify everything works:

```bash
bash TEST_WORKFLOW.sh
```

**What this does:**
- Creates a test dataset (50 samples)
- Trains a model (GPT-2, safe on any machine)
- Generates predictions
- Evaluates and shows results

**Time:** 8-15 minutes

**Expected output:** You'll see accuracy metrics confirming the pipeline works.

### Step 5: Train on Paper Evaluation Dataset

Train the model on the actual paper dataset:

```bash
# Make sure virtual environment is activated
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train the model
python3 -c "
from src.hybrid_dpo import train_hybrid_dpo
train_hybrid_dpo({
    'data': {
        'train_path': 'data/paper_eval/train.jsonl',
        'eval_path': 'data/paper_eval/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/paper_eval_model',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 1,
        'learning_rate': 5e-6,
        'gradient_accumulation_steps': 2
    }
})
"
```

**What this does:**
- Trains the model using DPO (Direct Preference Optimization)
- Saves checkpoints to `outputs/paper_eval_model/`
- Shows training progress in the terminal

**Time:** 10-20 minutes

### Step 6: Generate Predictions

Use the trained model to generate predictions on the test set:

```bash
python3 scripts/generate_predictions.py \
  --model_path outputs/paper_eval_model \
  --test_jsonl data/paper_eval/test.jsonl \
  --output_jsonl outputs/paper_eval_predictions.jsonl \
  --max_new_tokens 20
```

**What this does:**
- Loads the trained model
- Generates answers for test questions
- Saves predictions to a file

**Time:** 1-2 minutes

### Step 7: Evaluate Results

Compute evaluation metrics:

```bash
python3 scripts/comprehensive_eval.py \
  --gold_jsonl data/paper_eval/test.jsonl \
  --pred_jsonl outputs/paper_eval_predictions.jsonl \
  --output_json outputs/paper_eval_results.json
```

**What this does:**
- Compares predictions to correct answers
- Computes accuracy, precision, recall, F1
- Saves detailed metrics to JSON file

**Time:** < 1 minute

### Step 8: View Results

View the evaluation results:

```bash
cat outputs/paper_eval_results.json | python3 -m json.tool
```

Or open `outputs/paper_eval_results.json` in any text editor.

---

## Summary

**Complete workflow (copy-paste all at once):**

```bash
# Step 1: Clone (already done if you're reading this)
cd clean_repo

# Step 2: Setup
bash QUICK_START.sh

# Step 3: Generate datasets
bash scripts/generate_all_datasets.sh

# Step 4: Test (optional but recommended)
bash TEST_WORKFLOW.sh

# Step 5: Train (activate venv first)
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 -c "from src.hybrid_dpo import train_hybrid_dpo; train_hybrid_dpo({'data': {'train_path': 'data/paper_eval/train.jsonl', 'eval_path': 'data/paper_eval/val.jsonl'}, 'dpo': {'output_dir': 'outputs/paper_eval_model', 'num_train_epochs': 3, 'per_device_train_batch_size': 1, 'learning_rate': 5e-6, 'gradient_accumulation_steps': 2}})"

# Step 6: Generate predictions
python3 scripts/generate_predictions.py --model_path outputs/paper_eval_model --test_jsonl data/paper_eval/test.jsonl --output_jsonl outputs/paper_eval_predictions.jsonl --max_new_tokens 20

# Step 7: Evaluate
python3 scripts/comprehensive_eval.py --gold_jsonl data/paper_eval/test.jsonl --pred_jsonl outputs/paper_eval_predictions.jsonl --output_json outputs/paper_eval_results.json

# Step 8: View results
cat outputs/paper_eval_results.json | python3 -m json.tool
```

---

**Need more details?** See sections below for explanations of each component.

## Overview

This project combines:
- **Knowledge Graph Data Processing**: Prepare datasets from KG triples with optional visual graph rendering
- **DPO Training**: Fine-tune LLMs using Direct Preference Optimization on hybrid KG reasoning examples
- **Evaluation**: Scripts for link prediction and multi-hop QA evaluation

## Prerequisites

### System Requirements

**Minimum (for testing with GPT-2):**
- Python 3.10
- 4GB RAM
- CPU-only training works

**Recommended (for larger models):**
- Python 3.10
- GPU with 16GB+ VRAM (for models like Mistral-7B)
- OR 16GB+ RAM for CPU training

### System Dependencies

**Required:**
- Git
- Graphviz (for graph visualization)

**Install Graphviz:**

- **macOS**: 
  ```bash
  brew install graphviz
  ```

- **Ubuntu/Debian**: 
  ```bash
  sudo apt-get install graphviz
  ```

- **Windows**: Download from [Graphviz website](https://graphviz.org/download/) or use `choco install graphviz`

## Manual Installation (Optional)

**Note:** The `QUICK_START.sh` script (Step 2 in Quick Start) handles all setup automatically. Only follow these steps if you prefer manual setup.

### Prerequisites

- Python 3.10 or higher
- Git
- Graphviz (install via `brew install graphviz` on macOS, `apt-get install graphviz` on Linux)

### Setup Steps

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Set PYTHONPATH:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

4. **Verify setup:**
   ```bash
   python3 verify_setup.py
   ```

## Dataset Generation

### Generating All Dataset Variants

To reproduce all datasets used in the experiments, run:

```bash
bash scripts/generate_all_datasets.sh
```

This script generates:
- **data/hybrid/**: Basic dataset (50 samples, 80/10/10 split)
- **data/hybrid_large/**: Larger dataset (100 samples, 80/10/10 split)
- **data/hybrid_simcse/**: With SimCSE-based neighbor ranking (threshold 0.8)
- **data/hybrid_simcse_default/**: With SimCSE ranking using default settings
- **data/paper_eval/**: Final evaluation dataset (200 samples, 70/15/15 split, no images)

### Generating Individual Datasets

You can also generate datasets individually:

#### Basic Hybrid Dataset

```bash
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 50 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

#### Hybrid with SimCSE Ranking

```bash
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid_simcse \
  --limit 50 \
  --use_sns \
  --entity_texts_jsonl data/entity_texts.jsonl \
  --sns_top_k 5 \
  --sns_threshold 0.8 \
  --seed 42
```

#### Paper Evaluation Dataset

```bash
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/paper_eval \
  --limit 200 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --no_images
```

#### PrimeKG Dataset Processing

For full PrimeKG experiments:

```bash
# 1. Download PrimeKG
python scripts/primekg_download.py --target_dir third_party/PrimeKG

# 2. Create subset (optional)
python scripts/primekg_subset.py \
  --primekg_dir third_party/PrimeKG \
  --out_dir data/primekg \
  --limit_nodes 50000

# 3. Convert to JSONL
python scripts/primekg_convert_from_kgcsv.py \
  --kgcsv_path data/primekg/kg.csv \
  --out_jsonl data/primekg_triples.jsonl

# 4. Prepare hybrid dataset
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/primekg_triples.jsonl \
  --out_dir data/primekg/hybrid_minilm_smoke \
  --limit 1000
```

## Detailed Usage

### 1. Prepare Dataset

The `prepare_hybrid_dataset.py` script creates train/val/test splits from KG triples:

```bash
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl <path_to_triples.jsonl> \
  --out_dir <output_directory> \
  --limit <number_of_samples> \
  [--train_ratio 0.8] \
  [--val_ratio 0.1] \
  [--test_ratio 0.1] \
  [--seed 42] \
  [--no_images] \
  [--use_sns] \
  [--entity_texts_jsonl <path_to_entity_texts.jsonl>]
```

**Arguments:**
- `--triples_jsonl`: Path to input triples file (JSONL format: `{"head": "...", "relation": "...", "tail": "..."}`)
- `--out_dir`: Output directory (will contain `train.jsonl`, `val.jsonl`, `test.jsonl`, and optionally `images/`)
- `--limit`: Maximum number of samples to process
- `--train_ratio`, `--val_ratio`, `--test_ratio`: Split ratios (must sum to 1.0)
- `--seed`: Random seed for reproducibility
- `--no_images`: Skip graph image rendering (faster)
- `--use_sns`: Use SimCSE-based neighbor ranking
- `--entity_texts_jsonl`: Path to entity text descriptions (required if `--use_sns`)

**Output:**
- `train.jsonl`, `val.jsonl`, `test.jsonl`: Dataset splits
- `images/`: Rendered graph visualizations (if `--no_images` not used)

### 2. Train Model

#### Python API (Recommended)

```python
from src.hybrid_dpo import train_hybrid_dpo

# Basic usage (GPT-2, safe on any machine)
train_hybrid_dpo({
    "data": {
        "train_path": "data/hybrid/train.jsonl",
        "eval_path": "data/hybrid/val.jsonl"
    },
    "dpo": {
        "output_dir": "outputs/hybrid-dpo",
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "learning_rate": 5e-6
    }
})

# Using large models (requires GPU or 16GB+ RAM)
train_hybrid_dpo({
    "model": {
        "base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "data": {
        "train_path": "data/hybrid/train.jsonl",
        "eval_path": "data/hybrid/val.jsonl"
    },
    "dpo": {
        "output_dir": "outputs/hybrid-dpo",
        "beta": 0.5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4
    }
})
```

#### Shell Script (Linux/macOS)

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRAIN_JSONL="data/hybrid/train.jsonl"
export EVAL_JSONL="data/hybrid/val.jsonl"
export OUTPUT_DIR="outputs/hybrid-dpo"
export EPOCHS=2
export BSZ=4
export LR=5e-6

bash scripts/train_hybrid_dpo.sh
```

#### PowerShell (Windows)

```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
$env:TRAIN_JSONL = "data/hybrid/train.jsonl"
$env:EVAL_JSONL = "data/hybrid/val.jsonl"
$env:OUTPUT_DIR = "outputs/hybrid-dpo"
$env:EPOCHS = 2
$env:BSZ = 4
$env:LR = 5e-6

pwsh scripts/train_hybrid_dpo.ps1
```

**Training Parameters:**
- `output_dir`: Where to save checkpoints and logs
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `learning_rate`: Learning rate (default: 5e-6)
- `beta`: DPO beta parameter (default: 0.1)
- `gradient_accumulation_steps`: Gradient accumulation steps
- `logging_steps`: How often to log
- `save_steps`: How often to save checkpoints
- `deepspeed`: Path to DeepSpeed config (optional, for large models)

**Output:**
- Checkpoints saved in `output_dir/checkpoint-*/`
- Training logs and metrics

### 3. Generate Predictions

```bash
python scripts/generate_predictions.py \
  --model_path <path_to_trained_model> \
  --test_jsonl <path_to_test_data> \
  --output_jsonl <path_to_output_predictions> \
  [--max_new_tokens 20] \
  [--batch_size 8]
```

**Arguments:**
- `--model_path`: Path to trained model directory (contains `config.json`, `model.safetensors`, etc.)
- `--test_jsonl`: Path to test dataset
- `--output_jsonl`: Where to save predictions
- `--max_new_tokens`: Maximum tokens to generate
- `--batch_size`: Batch size for inference

**Output:**
- JSONL file with predictions (one per line)

### 4. Evaluate Results

#### Multi-hop QA Evaluation

```bash
python scripts/eval_multihop_qa.py \
  --gold_jsonl <path_to_gold_answers> \
  --pred_jsonl <path_to_predictions> \
  [--output_json <path_to_output_metrics>]
```

**Output:** Accuracy metric (printed to stdout and optionally saved to JSON)

#### Link Prediction Evaluation

```bash
python scripts/eval_link_prediction.py \
  --triples_jsonl <path_to_test_triples> \
  --predictions_jsonl <path_to_predictions> \
  [--output_json <path_to_output_metrics>]
```

**Output:** MRR and Hits@10 metrics (printed to stdout and optionally saved to JSON)

#### Comprehensive Evaluation

```bash
python scripts/comprehensive_eval.py \
  --gold_jsonl <path_to_gold_answers> \
  --pred_jsonl <path_to_predictions> \
  --output_json <path_to_output_metrics>
```

**Output:** Detailed metrics including accuracy, precision, recall, F1, and per-relation statistics

## Configuration

Configuration is managed through `src/config.py`. Key settings can be overridden when calling `train_hybrid_dpo()`:

- **Model**: `base_model_name_or_path` (default: `"gpt2"`)
- **Data**: `train_path`, `eval_path`
- **DPO**: `beta`, `num_train_epochs`, `learning_rate`, etc.
- **SNS**: `similarity_threshold`, `top_k` (for SimCSE ranking)

See `src/config.py` for all available options.

## Project Structure

```
.
├── src/                    # Core source code
│   ├── config.py          # Configuration management
│   ├── hybrid_dpo.py      # DPO training entrypoint
│   ├── kg_data.py         # KG data loading utilities
│   ├── prompting.py       # Prompt templates
│   ├── sns_ranker.py      # SimCSE-based ranking
│   ├── kg_visualize.py    # Graph visualization
│   └── ...
├── scripts/               # Executable scripts
│   ├── prepare_hybrid_dataset.py    # Main dataset preparation
│   ├── generate_all_datasets.sh     # Generate all dataset variants
│   ├── train_hybrid_dpo.sh          # Training launcher
│   ├── generate_predictions.py      # Generate model predictions
│   ├── eval_multihop_qa.py          # Multi-hop QA evaluation
│   ├── eval_link_prediction.py      # Link prediction evaluation
│   ├── comprehensive_eval.py        # Comprehensive evaluation
│   ├── primekg_download.py          # Download PrimeKG
│   ├── primekg_subset.py            # Create PrimeKG subset
│   ├── primekg_convert_from_kgcsv.py # Convert PrimeKG to JSONL
│   └── ...
├── data/                  # Data files
│   ├── sample_triples.jsonl    # Sample KG triples (included)
│   ├── entity_texts.jsonl      # Entity text descriptions (included)
│   ├── hybrid/                 # Generated: basic dataset
│   ├── hybrid_large/           # Generated: larger dataset
│   ├── hybrid_simcse/          # Generated: with SimCSE ranking
│   ├── hybrid_simcse_default/  # Generated: SimCSE default
│   ├── paper_eval/            # Generated: final evaluation dataset
│   └── primekg/               # Generated: PrimeKG datasets (if using PrimeKG)
├── gita_module/          # DeepSpeed configurations
├── requirements.txt      # Python dependencies
├── verify_setup.py      # Setup verification script
└── README.md            # This file
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Graphviz Errors

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'dot'`

**Solution:** Install Graphviz system package (see Prerequisites section).

### CUDA/GPU Issues

**Problem:** CUDA out of memory or GPU not detected

**Solution:**
- Reduce batch size: `"per_device_train_batch_size": 2`
- Use CPU: Set device to `"cpu"` in config
- Use gradient accumulation: Increase `"gradient_accumulation_steps"`

### Missing Data Files

**Problem:** `FileNotFoundError: data/sample_triples.jsonl`

**Solution:** The sample files should be included in the repository. If missing, check that you cloned the full repository.

### Model Download Issues

**Problem:** Hugging Face model download fails

**Solution:**
- Check internet connection
- Set `HF_HOME` environment variable if using custom cache location
- For large models, consider using `huggingface-cli` to download manually

## Reproducing Results

### For TAs/Reviewers: Complete Reproduction Workflow

To reproduce all results from the paper:

```bash
# 1. Verify setup
python verify_setup.py

# 2. Generate all dataset variants
bash scripts/generate_all_datasets.sh

# 3. Train on paper evaluation dataset
python -c "
from src.hybrid_dpo import train_hybrid_dpo
train_hybrid_dpo({
    'data': {
        'train_path': 'data/paper_eval/train.jsonl',
        'eval_path': 'data/paper_eval/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/paper_eval_model',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 1,
        'learning_rate': 5e-6,
        'gradient_accumulation_steps': 2
    }
})
"

# 4. Generate predictions
python scripts/generate_predictions.py \
  --model_path outputs/paper_eval_model \
  --test_jsonl data/paper_eval/test.jsonl \
  --output_jsonl outputs/paper_eval_predictions.jsonl

# 5. Comprehensive evaluation
python scripts/comprehensive_eval.py \
  --gold_jsonl data/paper_eval/test.jsonl \
  --pred_jsonl outputs/paper_eval_predictions.jsonl \
  --output_json outputs/paper_eval_results.json
```

## Example Workflow

Complete end-to-end example with basic dataset:

```bash
# 1. Verify setup
python verify_setup.py

# 2. Prepare dataset
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 50

# 3. Train model
python -c "
from src.hybrid_dpo import train_hybrid_dpo
train_hybrid_dpo({
    'data': {
        'train_path': 'data/hybrid/train.jsonl',
        'eval_path': 'data/hybrid/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/my_model',
        'num_train_epochs': 2,
        'per_device_train_batch_size': 4,
        'learning_rate': 5e-6
    }
})
"

# 4. Generate predictions
python scripts/generate_predictions.py \
  --model_path outputs/my_model \
  --test_jsonl data/hybrid/test.jsonl \
  --output_jsonl outputs/my_predictions.jsonl

# 5. Evaluate
python scripts/eval_multihop_qa.py \
  --gold_jsonl data/hybrid/test.jsonl \
  --pred_jsonl outputs/my_predictions.jsonl
```

## References

This project builds upon:
- [SNS](https://github.com/ruili33/SNS)
- [GITA](https://github.com/WEIYanbin1999/GITA)
- [GraphWiz](https://github.com/Graph-Reasoning-LLM)

## License

See individual third-party licenses in respective directories.

## Contact

For issues or questions, please open an issue on the repository.

