import os
from typing import List
from gensim.models.word2vec import Word2Vec
import json 
from pathlib import Path 
from tqdm import tqdm 
from gensim.models.callbacks import CallbackAny2Vec 

# --- Configuration ---
# Word2Vec Model Parameters similar to Devign
W2V_MIN_COUNT = 3       
W2V_VECTOR_SIZE = 100   
W2V_WINDOW = 5          
W2V_SG = 1              
W2V_HS = 0              
W2V_NEGATIVE = 5        
W2V_ALPHA = 0.01        
W2V_SAMPLE = 1e-5       
W2V_WORKERS = 4         
W2V_EPOCHS = 10         

# Output Configuration
MODEL_DIR = "models" 
MODEL_NAME = "custom_word2vec.model" 

# Data Configuration
TRAIN_DATA_FILENAME = "tokenized_functions.jsonl" 
TRAIN_DATA_SUBDIRS = ["data", "train"]

# Callback for TQDM progress bar during Word2Vec training
class TqdmWord2VecCallback(CallbackAny2Vec):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self, model):
        self.pbar = tqdm(total=self.total_epochs, desc="Word2Vec Training Progress (Epochs)")

    def on_epoch_end(self, model):
        if self.pbar:
            self.pbar.update(1)

    def on_train_end(self, model):
        if self.pbar:
            self.pbar.close()

def load_tokenized_corpus_from_jsonl(filepath: Path) -> List[List[str]]: 
    print(f"Loading tokenized corpus from {filepath}...")
    corpus = []
    if not filepath.is_file():
        print(f"Error: Training data file not found at {filepath}")
        return corpus

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading tokenized functions from JSONL"):
            try:
                tokens = json.loads(line)
                if isinstance(tokens, list) and all(isinstance(token, str) for token in tokens):
                    corpus.append(tokens)
                else:
                    print(f"Skipping line due to unexpected format (not a list of strings): {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decoding error: {e} - Line content: {line.strip()}")
            except Exception as e:
                print(f"An unexpected error occurred while processing line: {e} - Line: {line.strip()}")
    print(f"Loaded {len(corpus)} tokenized functions.")
    return corpus

def train_word2vec_on_custom_tokens(corpus_of_token_lists: List[List[str]], model_save_path: str): 
    """
    Trains a Word2Vec model on a pre-tokenized corpus.

    Args:
        corpus_of_token_lists: A list of lists of strings, where each inner list is the tokens of a document/function.
        model_save_path: The full path where the trained Word2Vec model will be saved.

    Returns:
        The trained gensim.models.word2vec.Word2Vec model, or None if training failed.
    """
    if not corpus_of_token_lists:
        print("No token lists provided. Aborting Word2Vec training.")
        return None

    print(f"Received {len(corpus_of_token_lists)} token lists for training.")
    print("Initializing Word2Vec model...")

    w2v_model = Word2Vec(
        min_count=W2V_MIN_COUNT,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        sg=W2V_SG,
        hs=W2V_HS,
        negative=W2V_NEGATIVE if W2V_HS == 0 else 0, # Use negative sampling only if HS is disabled
        alpha=W2V_ALPHA,
        sample=W2V_SAMPLE,
        workers=W2V_WORKERS,
        seed=42 # for reproducibility if needed
    )

    print("Building vocabulary from tokenized corpus...")
    w2v_model.build_vocab(corpus_iterable=corpus_of_token_lists, progress_per=1000)

    print(f"Vocabulary built. Size: {len(w2v_model.wv.index_to_key)} words.")
    print("Training Word2Vec model...")

    # Initialize TQDM callback
    tqdm_callback = TqdmWord2VecCallback(W2V_EPOCHS)

    w2v_model.train(
        corpus_iterable=corpus_of_token_lists,
        total_examples=w2v_model.corpus_count,
        epochs=W2V_EPOCHS, # Use W2V_EPOCHS directly as it's already defined
        report_delay=1,
        callbacks=[tqdm_callback] # Added TQDM callback
    )

    # Ensure the directory for saving the model exists
    model_dir = os.path.dirname(model_save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    print(f"Saving Word2Vec model to {model_save_path}...")
    w2v_model.save(model_save_path)
    print("Word2Vec model saved successfully.")
    return w2v_model

if __name__ == "__main__":
    print("Starting custom Word2Vec training process...")

    script_dir = Path(__file__).resolve().parent
    
    current_script_dir = Path(__file__).resolve().parent
    data_path_parts = TRAIN_DATA_SUBDIRS + [TRAIN_DATA_FILENAME]
    training_data_file_path = current_script_dir.joinpath(*data_path_parts)


    print(f"Attempting to load tokenized training data from: {training_data_file_path}")
    
    all_tokens_corpus = load_tokenized_corpus_from_jsonl(training_data_file_path)

    if not all_tokens_corpus: 
        print("No code data loaded. Please check the path and content of your training file.")
        print("Exiting Word2Vec training process.")
    else:
        print(f"Successfully loaded {len(all_tokens_corpus)} pre-tokenized code snippets for training.")
        
        output_model_dir = script_dir / MODEL_DIR
        final_model_path = output_model_dir / MODEL_NAME

        print(f"Output model will be saved to: {final_model_path}")

        trained_w2v_model = train_word2vec_on_custom_tokens(all_tokens_corpus, str(final_model_path))

        if trained_w2v_model:
            print("\\nWord2Vec training process completed.")
            print(f"Model saved at: {final_model_path}")
        else:
            print("\\nWord2Vec training process failed or no data was available to train on.") 