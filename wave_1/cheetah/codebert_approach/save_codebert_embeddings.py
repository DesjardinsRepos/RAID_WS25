import torch
from transformers import AutoTokenizer, AutoModel
import sys
import numpy as np
import json
from typing import List
from tqdm import tqdm
import os

# Global variables to keep loaded model and tokenizer in memory
_model = None
_tokenizer = None
_device = None

def load_model_and_tokenizer(model_name="microsoft/codebert-base"):
    """
    Load and cache the model and tokenizer
    """
    global _model, _tokenizer, _device
    
    if _model is None or _tokenizer is None:
        # Set device
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model {model_name} on {_device}...")
        
        # Load tokenizer and model (cached for subsequent calls)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(_device)
    
    return _model, _tokenizer, _device

def get_embedding(code_text, model_name="microsoft/codebert-base", cls_only=True):
    """
    Get CodeBERT embedding for a code snippet.
    
    Args:
        code_text: The code snippet as a string
        model_name: Name of the CodeBERT model to use
        cls_only: Whether to return only the [CLS] token embedding
        
    Returns:
        The code embedding as a numpy array
    """
    # Load or get cached model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    # Tokenize the code
    code_tokens = tokenizer.tokenize(code_text)
    
    # Add special tokens
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    
    # Convert tokens to ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Truncate if too long
    if len(token_ids) > 512:
        print("Warning: Code is too long, truncating to 512 tokens", file=sys.stderr)
        token_ids = token_ids[:511] + [tokenizer.eos_token_id]
    
    # Get embeddings - using exact syntax from the example
    token_ids_tensor = torch.tensor(token_ids).to(device)
    with torch.no_grad():
        context_embeddings = model(token_ids_tensor[None,:])[0]
    
    # Return only the [CLS] token embedding if requested
    if cls_only:
        return context_embeddings[0, 0].cpu().numpy()
    else:
        return context_embeddings[0].cpu().numpy()

def calc_features(functions: List[str], model_name="microsoft/codebert-base", batch_size=32) -> List[List[float]]:
    """
    Calculate CodeBERT embeddings for a list of functions.
    
    Args:
        functions: List of function strings
        model_name: Name of the CodeBERT model to use
        batch_size: Batch size for processing (to avoid memory issues)
        
    Returns:
        List of embeddings for each function
    """
    # Load or get cached model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    # Prepare result container
    embeddings = []
    
    # Process functions in batches
    for i in tqdm(range(0, len(functions), batch_size), desc="Calculating embeddings"):
        batch = functions[i:i+batch_size]
        batch_embeddings = []
        
        # Process each function individually but in a batch
        for func in batch:
            # Tokenize the code
            code_tokens = tokenizer.tokenize(func)
            
            # Add special tokens
            tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
            
            # Convert tokens to ids
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Truncate if too long
            if len(token_ids) > 512:
                token_ids = token_ids[:511] + [tokenizer.eos_token_id]
            
            # Get embeddings exactly as in the example
            with torch.no_grad():
                # Important: Using the exact same syntax as the example
                context_embeddings = model(torch.tensor(token_ids).to(device)[None,:])[0]
            
            # Extract the CLS token embedding
            embedding = context_embeddings[0, 0].cpu().numpy().tolist()
            batch_embeddings.append(embedding)
        
        embeddings.extend(batch_embeddings)
    
    return embeddings

def load_functions(filepath):
    """
    Load functions from a JSON lines file.
    
    Args:
        filepath: Path to the JSON lines file
        
    Returns:
        List of function strings
    """
    print(f"Loading functions from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        functions = [json.loads(line)['function'] for line in tqdm(f.readlines(), desc="Loading functions")]
    return functions

def save_embeddings(embeddings, output_path):
    """
    Save embeddings to a file.
    
    Args:
        embeddings: List of embeddings
        output_path: Path to save the embeddings
    """
    print(f"Saving {len(embeddings)} embeddings to {output_path}...")
    np.save(output_path, np.array(embeddings))
    print(f"Saved embeddings with shape {np.array(embeddings).shape}")

def main():
    """
    Process functions from data/train/functions.jsonl and save the embeddings
    """
    # Define input and output paths
    input_path = 'data/validation/functions.jsonl'
    output_path = 'data/validation/embeddings.npy'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load functions
    functions = load_functions(input_path)
    print(f"Loaded {len(functions)} functions")
    
    # Calculate embeddings
    embeddings = calc_features(functions)
    
    # Save embeddings
    save_embeddings(embeddings, output_path)
    
    print("Done!")

if __name__ == "__main__":
    main() 