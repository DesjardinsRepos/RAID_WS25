import os
import json
from typing import List
from pathlib import Path
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
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


def calc_features(functions: List[str]) -> List[List[float]]:
    """
    Tokenizes a list of raw function strings, converts tokens to vectors using a
    pre-trained Word2Vec model (loaded from a default path), and computes a feature vector
    (mean of token vectors) for each function.

    Args:
        functions: A list of raw C/C++ function strings.

    Returns:
        A list of feature vectors. Each feature vector is a list of floats,
        representing the mean of token embeddings for a function. Returns a
        zero vector if a function has no tokens in the model's vocabulary.
    """
    try:
        w2v_model = load_w2v_model() 
    except FileNotFoundError as e:
        print(e)
        return [[] for _ in functions] 
    
    vector_size = w2v_model.vector_size

    all_function_feature_vectors: List[List[float]] = []

    print("Calculating features for functions...")
    for func_str in tqdm(functions, desc="Processing functions"):
        try:
            tokens = tokenize_code(func_str)
        except Exception as e:
            print(f"Error tokenizing function: {e}. Using zero vector.")
            tokens = [] # Proceed with empty tokens to generate a zero vector

        token_vectors_for_current_func = []
        if tokens: # Check if tokenization produced any tokens
            for token in tokens:
                if token in w2v_model.wv:
                    token_vectors_for_current_func.append(w2v_model.wv[token])
        
        if token_vectors_for_current_func:
            mean_vector = np.mean(token_vectors_for_current_func, axis=0).tolist()
            all_function_feature_vectors.append(mean_vector)
        else:
            # Append a zero vector if no tokens were found in vocab or function was empty/untokenizable
            all_function_feature_vectors.append(np.zeros(vector_size).tolist())
        if not tokens:
                    pass # Already printed error or function was empty
               


    print(f"Feature calculation complete. Generated {len(all_function_feature_vectors)} feature vectors.")
    return all_function_feature_vectors


