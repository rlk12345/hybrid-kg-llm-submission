#!/usr/bin/env python3
"""
Comprehensive evaluation script for research paper.
Computes multiple metrics: accuracy, precision, recall, F1, per-relation stats.
"""
import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

def accuracy(preds: List[str], golds: List[str]) -> float:
    """Compute accuracy."""
    correct = sum(1 for p, g in zip(preds, golds) if p.strip().lower() == g.strip().lower())
    return correct / max(1, len(golds))

def precision_recall_f1(preds: List[str], golds: List[str]) -> Tuple[float, float, float]:
    """Compute precision, recall, F1."""
    # For multi-class, compute macro-averaged metrics
    relation_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred, gold in zip(preds, golds):
        pred = pred.strip().lower()
        gold = gold.strip().lower()
        
        if pred == gold:
            relation_stats[gold]['tp'] += 1
        else:
            relation_stats[gold]['fn'] += 1
            relation_stats[pred]['fp'] += 1
    
    precisions = []
    recalls = []
    f1s = []
    
    all_relations = set(relation_stats.keys())
    for rel in all_relations:
        stats = relation_stats[rel]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-10, prec + rec)
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    macro_prec = sum(precisions) / max(1, len(precisions))
    macro_rec = sum(recalls) / max(1, len(recalls))
    macro_f1 = sum(f1s) / max(1, len(f1s))
    
    return macro_prec, macro_rec, macro_f1

def per_relation_accuracy(preds: List[str], golds: List[str]) -> Dict[str, float]:
    """Compute accuracy per relation type."""
    relation_correct = defaultdict(int)
    relation_total = defaultdict(int)
    
    for pred, gold in zip(preds, golds):
        gold_lower = gold.strip().lower()
        relation_total[gold_lower] += 1
        if pred.strip().lower() == gold_lower:
            relation_correct[gold_lower] += 1
    
    per_rel_acc = {}
    for rel in relation_total:
        per_rel_acc[rel] = relation_correct[rel] / relation_total[rel]
    
    return per_rel_acc

def error_analysis(preds: List[str], golds: List[str], prompts: List[str]) -> List[Dict]:
    """Identify error cases for analysis."""
    errors = []
    for pred, gold, prompt in zip(preds, golds, prompts):
        if pred.strip().lower() != gold.strip().lower():
            errors.append({
                'prediction': pred,
                'gold': gold,
                'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt
            })
    return errors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive evaluation")
    parser.add_argument("--gold_jsonl", type=str, required=True)
    parser.add_argument("--pred_jsonl", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None, help="Save detailed results to JSON")
    args = parser.parse_args()
    
    # Load data
    golds = []
    preds = []
    prompts = []
    
    with open(args.gold_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                golds.append(str(obj.get("answer") or obj.get("chosen", "")))
                prompts.append(obj.get("prompt", ""))
    
    with open(args.pred_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                preds.append(str(obj.get("prediction", "")))
    
    if len(preds) != len(golds):
        print(f"Error: Mismatch - {len(preds)} predictions vs {len(golds)} gold labels", file=sys.stderr)
        sys.exit(1)
    
    # Compute metrics
    acc = accuracy(preds, golds)
    prec, rec, f1 = precision_recall_f1(preds, golds)
    per_rel = per_relation_accuracy(preds, golds)
    errors = error_analysis(preds, golds, prompts)
    
    # Print results
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nDataset Size: {len(golds)} samples")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\nPer-Relation Accuracy:")
    for rel, rel_acc in sorted(per_rel.items()):
        count = sum(1 for g in golds if g.strip().lower() == rel)
        print(f"  {rel:20s}: {rel_acc:.4f} ({rel_acc*100:.2f}%) - {count} samples")
    
    print(f"\nError Analysis:")
    print(f"  Total Errors: {len(errors)}")
    print(f"  Error Rate: {len(errors)/len(golds):.4f} ({len(errors)/len(golds)*100:.2f}%)")
    
    if errors and len(errors) <= 10:
        print(f"\n  Sample Errors:")
        for i, err in enumerate(errors[:5], 1):
            print(f"    {i}. Predicted: '{err['prediction']}' | Gold: '{err['gold']}'")
    
    # Save detailed results
    results = {
        'overall': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'num_samples': len(golds),
            'num_errors': len(errors)
        },
        'per_relation': per_rel,
        'errors': errors[:20]  # Save first 20 errors
    }
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Detailed results saved to {args.output_json}")
    
    # Also print JSON for easy parsing
    print("\n" + "=" * 60)
    print("JSON Summary:")
    print(json.dumps({
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'num_samples': len(golds),
        'num_errors': len(errors)
    }, indent=2))

