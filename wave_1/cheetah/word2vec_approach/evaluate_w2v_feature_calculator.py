import os
import json
from typing import List
from pathlib import Path
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import sys

# Add the parent directory to Python path to allow imports
sys.path.append(str(Path(__file__).parent))
from code_tokenizer import tokenize_code


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_RAW_FUNCTIONS_FILENAME = "functions.jsonl" # Original, non-tokenized functions
DEFAULT_RAW_FUNCTIONS_SUBDIRS = ["data", "train"]
DEFAULT_RAW_FUNCTIONS_PATH = SCRIPT_DIR.joinpath(*DEFAULT_RAW_FUNCTIONS_SUBDIRS, DEFAULT_RAW_FUNCTIONS_FILENAME)


def load_w2v_model(model_path: str = 'models/custom_word2vec.model') -> Word2Vec:
    """Load a trained Word2Vec model from disk.

    Args:
        model_path: Path to the model file (e.g., 'models/custom_word2vec.model').
                    Relative to the execution directory or an absolute path.

    Returns:
        The loaded gensim.models.word2vec.Word2Vec model.

    Raises:
        FileNotFoundError: If the model file is not found.
    """
    path_obj = Path(model_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Word2Vec model file not found at {path_obj.resolve()}")
    
    print(f"Loading Word2Vec model from {path_obj.resolve()}...")
    loaded_model = Word2Vec.load(str(path_obj)) # Word2Vec.load expects a string path
    print(f"Word2Vec model '{path_obj.name}' loaded successfully. Vector size: {loaded_model.vector_size}")
    return loaded_model


def load_raw_functions_from_jsonl(filepath: Path) -> List[str]:
    """
    Loads raw function strings from a JSONL file.
    Each line in the file is expected to be a JSON object with a 'function' key.
    """
    print(f"Loading raw functions from {filepath}...")
    functions = []
    if not filepath.is_file():
        print(f"Error: Raw functions file not found at {filepath}")
        return functions

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading raw functions"):
            try:
                data = json.loads(line)
                if 'function' in data and isinstance(data['function'], str):
                    functions.append(data['function'])
                else:
                    print(f"Skipping line due to missing 'function' key or incorrect type: {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decoding error: {e} - Line content: {line.strip()}")
            except Exception as e:
                print(f"An unexpected error occurred while processing line: {e} - Line: {line.strip()}")
    print(f"Loaded {len(functions)} raw functions.")
    return functions


def calc_features_w2v(tokenized_functions_path: str = "data/train/tokenized_functions.jsonl") -> List[List[float]]:
    """
    Reads pre-tokenized functions from a JSONL file, converts tokens to vectors using a
    pre-trained Word2Vec model, and computes a feature vector (mean of token vectors) for each function.

    Args:
        tokenized_functions_path: Path to the JSONL file containing pre-tokenized functions.

    Returns:
        A list of feature vectors. Each feature vector is a list of floats,
        representing the mean of token embeddings for a function. Returns a
        zero vector if a function has no tokens in the model's vocabulary.
    """
    try:
        w2v_model = load_w2v_model() 
    except FileNotFoundError as e:
        print(e)
        return []
    
    vector_size = w2v_model.vector_size
    all_function_feature_vectors: List[List[float]] = []

    print("Calculating features for pre-tokenized functions...")
    with open(tokenized_functions_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing functions"):
            try:
                tokens = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}. Using zero vector.")
                all_function_feature_vectors.append(np.zeros(vector_size).tolist())
                continue

            token_vectors_for_current_func = []
            if tokens:  # Check if we have any tokens
                for token in tokens:
                    if token in w2v_model.wv:
                        token_vectors_for_current_func.append(w2v_model.wv[token])
            
            if token_vectors_for_current_func:
                mean_vector = np.mean(token_vectors_for_current_func, axis=0).tolist()
                all_function_feature_vectors.append(mean_vector)
            else:
                # Append a zero vector if no tokens were found in vocab
                all_function_feature_vectors.append(np.zeros(vector_size).tolist())

    print(f"Feature calculation complete. Generated {len(all_function_feature_vectors)} feature vectors.")
    return all_function_feature_vectors


