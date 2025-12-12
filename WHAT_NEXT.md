# What's Next? A Guide to Using This Repository

## I Just Ran `generate_all_datasets.sh` - What Happened?

The script created 5 different dataset variants in the `data/` directory:

1. **`data/hybrid/`** - Basic dataset (50 samples)
   - 80% train, 10% validation, 10% test
   - Standard dataset without special features

2. **`data/hybrid_large/`** - Larger dataset (100 samples)
   - Same structure as hybrid, but more data

3. **`data/hybrid_simcse/`** - With SimCSE ranking (50 samples)
   - Uses semantic similarity to select relevant KG neighbors
   - Helps the model focus on relevant information

4. **`data/hybrid_simcse_default/`** - SimCSE with default settings
   - Same as above but with default similarity thresholds

5. **`data/paper_eval/`** - Final evaluation dataset (200 samples)
   - 70% train, 15% validation, 15% test
   - Used for final paper results
   - No images (faster to generate)

**Each dataset contains:**
- `train.jsonl` - Training examples (DPO pairs)
- `val.jsonl` - Validation examples (for monitoring during training)
- `test.jsonl` - Test examples (for final evaluation)
- `images/` - Graph visualizations (optional, not in paper_eval)

## What Should I Do Next?

### Step 1: Test the Complete Pipeline (5-10 minutes)

**Run the test workflow to verify everything works:**

```bash
bash TEST_WORKFLOW.sh
```

**What this does:**
- Creates a small test dataset (20 samples)
- Trains a GPT-2 model (small, safe on any machine)
- Generates predictions
- Evaluates and shows results

**Why do this first?**
- Verifies your setup is correct
- Shows you the complete workflow
- Takes only 5-10 minutes
- Uses a small model that won't crash your system

### Step 2: Train on Paper Evaluation Dataset

Once you've verified the test works, train on the actual paper dataset:

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

**What's happening:**
- The model learns to prefer correct KG reasoning over incorrect answers
- Training uses DPO (Direct Preference Optimization)
- Checkpoints are saved in `outputs/paper_eval_model/`
- Training progress is logged to the console

**Time:** 10-20 minutes (depending on your machine)

### Step 3: Generate Predictions

After training, generate predictions on the test set:

```bash
python3 scripts/generate_predictions.py \
  --model_path outputs/paper_eval_model \
  --test_jsonl data/paper_eval/test.jsonl \
  --output_jsonl outputs/paper_eval_predictions.jsonl \
  --max_new_tokens 20
```

**What's happening:**
- The trained model reads test questions
- Generates answers (predictions)
- Saves predictions to a JSONL file

**Time:** 1-2 minutes

### Step 4: Evaluate Results

Finally, evaluate how well the model performed:

```bash
python3 scripts/comprehensive_eval.py \
  --gold_jsonl data/paper_eval/test.jsonl \
  --pred_jsonl outputs/paper_eval_predictions.jsonl \
  --output_json outputs/paper_eval_results.json
```

**What's happening:**
- Compares predictions to correct answers
- Computes accuracy, precision, recall, F1
- Saves detailed metrics to JSON file

**Output:** You'll see metrics like:
- Accuracy: X%
- Precision: X%
- Recall: X%
- F1 Score: X%

## Understanding the Outputs

### Dataset Files (`data/*/`)

Each dataset directory contains:
- **`train.jsonl`** - One example per line, each with:
  - `prompt`: The question and KG context
  - `chosen`: The correct answer (what model should learn)
  - `rejected`: The incorrect answer (what model should avoid)
  - `image`: Path to graph visualization (if generated)

### Model Checkpoints (`outputs/*/`)

After training, you'll find:
- `checkpoint-*/` - Model checkpoints at different training steps
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer files
- Training logs and metrics

### Predictions (`outputs/*_predictions.jsonl`)

Each line contains:
- `prompt`: The test question
- `prediction`: What the model predicted
- Other metadata

### Results (`outputs/*_results.json`)

JSON file with:
- `accuracy`: Overall accuracy
- `precision`, `recall`, `f1`: Detailed metrics
- Per-relation statistics (if applicable)

## Common Next Steps

### Experiment with Different Datasets

Try training on different dataset variants:

```bash
# Train on hybrid_large (more data)
python3 -c "
from src.hybrid_dpo import train_hybrid_dpo
train_hybrid_dpo({
    'data': {
        'train_path': 'data/hybrid_large/train.jsonl',
        'eval_path': 'data/hybrid_large/val.jsonl'
    },
    'dpo': {'output_dir': 'outputs/hybrid_large_model', 'num_train_epochs': 2}
})
"
```

### Compare Different Models

You can train with different base models (requires GPU for large models):

```bash
python3 -c "
from src.hybrid_dpo import train_hybrid_dpo
train_hybrid_dpo({
    'model': {'base_model_name_or_path': 'mistralai/Mistral-7B-Instruct-v0.2'},
    'data': {
        'train_path': 'data/paper_eval/train.jsonl',
        'eval_path': 'data/paper_eval/val.jsonl'
    },
    'dpo': {'output_dir': 'outputs/mistral_model', 'num_train_epochs': 2}
})
"
```

**Warning:** Large models (7B+ parameters) require GPU or 16GB+ RAM.

### Visualize Results

Check individual predictions:

```bash
# Look at first few predictions
head -5 outputs/paper_eval_predictions.jsonl | python3 -m json.tool
```

### Explore the Code

Key files to understand:
- `src/hybrid_dpo.py` - Main training logic
- `src/prompting.py` - How prompts are formatted
- `src/kg_data.py` - How KG data is loaded
- `scripts/prepare_hybrid_dataset.py` - How datasets are created

## Troubleshooting

### "Command not found: python"
- Use `python3` instead of `python` on macOS

### "ModuleNotFoundError: No module named 'src'"
- Set PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

### "CUDA out of memory"
- Reduce batch size: `'per_device_train_batch_size': 1`
- Use gradient accumulation: `'gradient_accumulation_steps': 4`
- Use CPU: Set device to `'cpu'` in config

### Training is too slow
- Use smaller model (GPT-2 is default)
- Reduce number of epochs
- Use `--no_images` when generating datasets

## Getting Help

- Check the main README.md for detailed documentation
- Review error messages - they usually point to the issue
- Verify setup: `python3 verify_setup.py`
- Check that you're in the `clean_repo` directory
- Make sure virtual environment is activated

## Summary: Quick Reference

```bash
# 1. Setup (one time)
bash QUICK_START.sh

# 2. Test everything works
bash TEST_WORKFLOW.sh

# 3. Generate all datasets
bash scripts/generate_all_datasets.sh

# 4. Train model
python3 -c "from src.hybrid_dpo import train_hybrid_dpo; train_hybrid_dpo({...})"

# 5. Generate predictions
python3 scripts/generate_predictions.py --model_path ... --test_jsonl ...

# 6. Evaluate
python3 scripts/comprehensive_eval.py --gold_jsonl ... --pred_jsonl ...
```

