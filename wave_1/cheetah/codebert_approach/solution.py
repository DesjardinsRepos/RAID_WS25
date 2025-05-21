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

def load_model_and_tokenizer(model_path="./codebert_local"):
    """
    Load and cache the model and tokenizer from a local path.
    """
    global _model, _tokenizer, _device
    
    if _model is None or _tokenizer is None:
        # Set device
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path} on {_device}...")
        
        # Load tokenizer and model (cached for subsequent calls)
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            _model = AutoModel.from_pretrained(model_path).to(_device)
        except Exception as e:
            print(f"Error loading model/tokenizer from {model_path}: {e}", file=sys.stderr)
            print("Please ensure you have downloaded the model to the specified path using the download_model.py script.", file=sys.stderr)
            sys.exit(1)
            
    return _model, _tokenizer, _device

def get_embedding(code_text, model_path="./codebert_local", cls_only=True):
    """
    Get CodeBERT embedding for a code snippet using a locally saved model.
    
    Args:
        code_text: The code snippet as a string
        model_path: Path to the locally saved CodeBERT model and tokenizer
        cls_only: Whether to return only the [CLS] token embedding
        
    Returns:
        The code embedding as a numpy array
    """
    # Load or get cached model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    
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

def calc_features(functions: List[str]) -> List[List[float]]:
    """
    Calculate CodeBERT embeddings for a list of functions.
    The model and tokenizer will be loaded from the default local path.
    
    Args:
        functions: List of function strings
        
    Returns:
        List of embeddings for each function
    """
    # Load or get cached model and tokenizer (will use default local path)
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Prepare result container
    embeddings = []
    
    # Process functions
    for func in tqdm(functions, desc="Calculating embeddings"):
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
        embeddings.append(embedding)
    
    return embeddings