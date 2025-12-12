#!/usr/bin/env python3
"""
Generate synthetic KG data for evaluation when real data is limited.
Creates diverse triples with multiple relation types.
"""
import json
import random
import argparse
from typing import List, Tuple

# Common medical/biomedical entities and relations
ENTITIES = [
    "Aspirin", "Ibuprofen", "Metformin", "Insulin", "Trastuzumab",
    "PTGS1", "PTGS2", "AMPK", "ERBB2", "INS", "GLUT4",
    "Headache", "Inflammation", "Type 2 Diabetes", "Breast Cancer",
    "Metabolism", "Glucose", "Protein", "Cell", "Tissue",
    "Gene A", "Gene B", "Protein X", "Protein Y", "Enzyme Z",
    "Disease A", "Disease B", "Symptom X", "Symptom Y",
    "Drug Alpha", "Drug Beta", "Drug Gamma", "Drug Delta",
    "Pathway 1", "Pathway 2", "Biological Process A", "Biological Process B"
]

RELATIONS = [
    "treats", "targets", "associated_with", "causes", "prevents",
    "inhibits", "activates", "regulates", "interacts_with", "binds_to",
    "metabolizes", "produces", "consumes", "upregulates", "downregulates"
]

def generate_triples(num_triples: int, seed: int = 42) -> List[Tuple[str, str, str]]:
    """Generate synthetic triples."""
    random.seed(seed)
    triples = []
    
    # Ensure we have enough entities
    entities = ENTITIES.copy()
    while len(entities) < num_triples * 2:
        entities.extend([f"Entity_{i}" for i in range(len(entities), len(entities) + 100)])
    
    for i in range(num_triples):
        head = random.choice(entities)
        relation = random.choice(RELATIONS)
        tail = random.choice(entities)
        
        # Avoid self-loops
        while tail == head:
            tail = random.choice(entities)
        
        triples.append((head, relation, tail))
    
    return triples

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic KG data")
    parser.add_argument("--output", type=str, default="data/synthetic_triples.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--num_triples", type=int, default=200,
                       help="Number of triples to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()
    
    print(f"Generating {args.num_triples} synthetic triples...")
    triples = generate_triples(args.num_triples, args.seed)
    
    # Write to JSONL
    with open(args.output, 'w', encoding='utf-8') as f:
        for h, r, t in triples:
            f.write(json.dumps({"head": h, "relation": r, "tail": t}, ensure_ascii=False) + "\n")
    
    print(f"✓ Generated {len(triples)} triples")
    print(f"  Saved to: {args.output}")
    
    # Show statistics
    relations = [r for _, r, _ in triples]
    unique_relations = set(relations)
    print(f"\nStatistics:")
    print(f"  Unique relations: {len(unique_relations)}")
    print(f"  Relation distribution:")
    for rel in sorted(unique_relations):
        count = relations.count(rel)
        print(f"    {rel}: {count}")

if __name__ == "__main__":
    main()

