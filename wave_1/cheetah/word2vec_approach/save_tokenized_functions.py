import os
import json
from code_tokenizer import tokenize_code

def save_tokenized_functions():
    # Process both training and validation sets
    datasets = {
        'validation': 'data/validation/functions.jsonl'
    }
    
    for dataset_name, jsonl_path in datasets.items():
        tokens_cache_path = jsonl_path.replace('functions.jsonl', 'tokenized_functions.jsonl')
        
        if os.path.exists(jsonl_path):
            print(f"\nProcessing {dataset_name} dataset...")
            samples = []
            with open(jsonl_path, encoding="utf-8") as fh:
                for line in fh:
                    src = json.loads(line)["function"]
                    samples.append(tokenize_code(src))
            
            # Save tokenized functions to cache
            print(f"Saving tokenized functions to cache {tokens_cache_path}...")
            os.makedirs(os.path.dirname(tokens_cache_path), exist_ok=True)
            with open(tokens_cache_path, 'w', encoding="utf-8") as fh:
                for tokens in samples:
                    fh.write(json.dumps(tokens) + "\n")
            
            print(f"Saved {len(samples)} tokenized functions to cache")
        else:
            print(f"Warning: Input file {jsonl_path} not found, skipping {dataset_name} dataset")

if __name__ == "__main__":
    save_tokenized_functions()