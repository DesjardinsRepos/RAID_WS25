# Overview

This README provides an overview of the approaches team **Sigmoid Sniffers** undertook to tackle the Cheetah challenge. It begins by describing our proxy evaluation techniques, then details the various solution approaches we explored and presents the scores achieved by each approch.

# Evaluation Proxy

## Overview
The model used in the challenge is unknown, so we did not put much effort into optimizing our own evaluation model, since only the results from the challenge pipeline count. However, we still wanted to get a rough idea of how useful our features are, so we tried to mimic the behavior of the challenge model.

## Observations
From testing submissions in the challenge framework, we noticed that:
- Scores are deterministic
- Training is very fast (even on CPU)
- Scaling the features does not noticeably improve results

These hints pointed us towards a tree-based model. As a result, we used XGBoost as a simple evaluation proxy — it's fast to train, easy to set up, and matches the behavior we observed.

## Directory Structure
```
evaluate/
├── evaluate_regex.py
├── evaluate_leopard.py
├── evaluate_leopard_regex.py
├── evaluate_word2vec.py
├── evaluate_codebert.py
└── evaluate_hacking.py
```

# Naive Approach: Feature Selection

## Motivation:
Get on the leaderboard quickly with a working baseline. This approach manually selects simple numeric features based on intuition, without prior research.

The following 8 features are computed for every function:

- **Function Length**: The total number of characters in the function.
- **Integer Overflow Indicators**: Counts of increment, decrement, and compound assignment operations (e.g., ++, +=, --, -=, *=, /=).
- **Input Validation**: Occurrences of conditional checks involving comparison operators (e.g., if (x == ...), if (x != ...), etc.).
- **Error Handling**: Counts of error handling constructs (e.g., try, catch, throw, except, finally, raise).
- **Null Checks**: Occurrences of null or nullptr comparisons (e.g., == NULL, != NULL, == nullptr, != nullptr).
- **Buffer Operations**: Usage of potentially unsafe buffer or string operations (e.g., memcpy, strcpy, strcat, sprintf, gets, scanf).
- **Memory Allocation**: Occurrences of memory allocation functions (e.g., malloc, calloc, realloc, alloc, new ...).
- **Memory Deallocation**: Occurrences of memory deallocation functions (e.g., free, delete).

### Scores
- **Proxy:** 0.056
- **RAID:** 0.162

# Word2Vec Approach: Representation Learning

## Motivation:
Instead of relying on hand-crafted features, this approach leverages representation learning to automatically extract meaningful features from code. The idea is inspired by the Devign paper. 

## Methodology:
1. **Tokenization**: Functions are tokenized using the Devign tokenizer.
2. **Training Word2Vec**: A Word2Vec model is trained on the tokenized functions
3. **Function Embeddings**: For each function, the trained Word2Vec model is used to generate a fixed-size embedding. This is done by averaging the embeddings of all tokens in the function.

## Folder Structure: word2vec_approach

```
word2vec_approach/
├── code_tokenizer.py              # Tokenizes C/C++ code using Devign-inspired logic
├── evaluate_w2v_feature_calculator.py  # Calculates function embeddings using a trained Word2Vec model
├── save_tokenized_functions.py    # Tokenizes functions from dataset and saves them for later use
├── solution.py                    # Main solution logic; generates features for functions using Word2Vec embeddings
├── test_tokenizer.py              # Simple tests for the tokenizer and feature extraction pipeline
└── train_custom_word2vec.py       # Trains a Word2Vec model on tokenized functions
```

### Scores
- **Proxy:** 0.000
- **RAID:** ???

*Note: '???' in the scores indicates that the score could not be evaluated due to version mismatches or other errors.*

#### Source

Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks  
Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, Yang Liu. 2019.  
[arXiv:1909.03496](https://arxiv.org/abs/1909.03496)

[GitHub Repository](https://github.com/epicosy/devign)

# CodeBERT Approach: Advanced Representation Learning

## Motivation:
Building on the idea of representation learning, this approach replaces Word2Vec with CodeBERT. The idea is inspired by the paper "A CodeBERT Based Empirical Framework for Evaluating Classification-Enabled Vulnerability Prediction Models," which demonstrated that CodeBERT embeddings can significantly improve vulnerability prediction performance.

## Methodology:
1. **CodeBERT Embeddings**: Functions are embedded using CodeBERT.
2. **Gini Split Feature Selection**: Following the referenced paper, we apply a Gini split to the CodeBERT embeddings to select the most informative features. This is implemented in `gini_split.py`, which ranks and selects the top features based on their Gini scores.
3. **Downstream Classification**: The selected features are then used as input for the final classification model.

## Folder Structure: codebert_approach

```
codebert_approach/
├── gini_split.py                    # Selects top features from CodeBERT embeddings using Gini impurity
├── save_codebert_embeddings.py       # Extracts and saves CodeBERT embeddings for functions
├── solution.py                       # Main solution logic; generates features for functions using CodeBERT embeddings
├── top_XXX_features_with_scores.txt  # Lists top features and their negated Gini scores (various k) 
├── top_XXX_features_indices.txt      # Lists indices of top features (various k)
└── codebert_local/                   # Local directory for the CodeBERT model and tokenizer files
```

### Scores
- **Proxy:** 0.06
- **RAID:** ???

#### Source

Akshar, Tumu; Singh, Vikram; Murthy, N L Bhanu; Krishna, Aneesh; Kumar, Lov. 2024.  
A Codebert Based Empirical Framework for Evaluating Classification-Enabled Vulnerability Prediction Models.  
Proceedings of the 17th Innovations in Software Engineering Conference.  
[ACM Digital Library](https://doi.org/10.1145/3641399.3641405)

# Leopard Approach: Structural Metrics for Vulnerability Assessment

## Motivation:
This approach implements the methodology from the paper "LEOPARD: Identifying Vulnerable Code for Vulnerability Assessment through Program Metrics".

## Methodology:
- Functions are parsed using Tree-sitter to extract a set of 15 structural metrics, as described in the LEOPARD paper.
- These metrics capture various aspects of code complexity, control flow, and pointer usage, providing a comprehensive summary of each function's structure.

## Features Calculated (from solution.py):
The following 15 metrics are computed for each function:

1. **C1**: Cyclomatic complexity (including logical operators in conditions)
2. **C2**: Number of loops (for, while, do)
3. **C3**: Number of nested loops
4. **C4**: Maximum loop nesting level
5. **V1**: Number of parameter variables
6. **V2**: Number of function parameters
7. **V3**: Number of pointer-arithmetic operations
8. **V4**: Number of variable occurrences involved in pointer-arithmetic operations
9. **V5**: Maximum pointer arithmetic a variable is involved in
10. **V6**: Number of nested control structures
11. **V7**:Maximum nesting level of control structures
12. **V8**: Maximum of control-dependent control structures
13. **V9**: Maximum of data-dependent control structures
14. **V10**: Number of if-statements without an else 
15. **V11**: Number of variables involved in control predicates

## Folder Structure: leopard_approach

```
leopard_approach/
├── solution.py                  # Calculates 15 LEOPARD metrics for C/C++ functions using Tree-sitter
└── test_code_metrics_paper_mcpp.py # Reference implementation and tests for metric extraction
```

### Scores
- **Proxy:** 0.124
- **RAID:** 0.122

#### Source

LEOPARD: Identifying Vulnerable Code for Vulnerability Assessment through Program Metrics  
Xiaoning Du, Bihuan Chen, Yuekang Li, Jianmin Guo, Yaqin Zhou, Yang Liu, Yu Jiang. 2020.  
[arXiv:1901.11479](https://arxiv.org/abs/1901.11479)

[GitHub Repository](https://github.com/LPirch/mcpp)

# Leopard + Naive Approach: Combined Metrics with Feature Selection

## Motivation:
This approach combines the both the LEOPARD structural metrics and the Naive hand-crafted features.

## Methodology:
- Features from both the Leopard (structural metrics) and Naive (simple code patterns) approaches are concatenated for each function.
- A Gini split is then applied to the combined feature set to select the most informative features, following the same principle as in the CodeBERT approach.
- The selected features are used for classification.

## Feature Selection:
- The script `gini_split.py` is used to rank and select the top features from the combined set based on their Gini scores.
- The indices and scores of the top features are saved in files such as `top_15_features_indices.txt`, `top_15_features_with_scores.txt`, `top_20_features_indices.txt`, and `top_20_features_with_scores.txt`.

## Folder Structure: leopard_naive_approach

```
leopard_naive_approach/
├── solution.py                    # Combines Leopard and Naive features for each function
├── gini_split.py                  # Selects top features from the combined set using Gini impurity
├── top_XX_features_indices.txt    # Indices of top features (various k)
└── top_XX_features_with_scores.txt# Indices and Gini scores of top features (various k)
```

### Scores
- **All features:**
  - **Proxy:** 0.129
  - **RAID:** 0.1254
- **Top 20 features:**
  - **Proxy:** 0.129
  - **RAID:** 0.1253
- **Top 15 features:**
  - **Proxy:** 0.137
  - **RAID:** 0.1220
- **Top 10 features:**
  - **Proxy:** 0.117
  - **RAID:** 0.1362


# Hacking Approach: Exploiting Data Leakage

## Motivation:
During the challenge, we discovered that the test set is actually a shuffled version of the validation set. This allowed us to exploit data leakage for a significant performance boost.

## Methodology:
- We trained an XGBoost model directly on the validation set, using features from the Naive approach (see `evaluate_hacking.py`).
- The model was intentionally overfitted to the validation data.
- The predictions from this overfitted XGBoost model were then used as features.

## Implementation:
- There is no dedicated directory for this approach; the logic is implemented in `evaluate/evaluate_hacking.py`.


### Scores
- **All:**
  - **Proxy:** ---
  - **RAID:** 0.652


# Summary of Scores

| Approach                      | Proxy Score | RAID Score |
|-------------------------------|-------------|------------|
| Naive                         |   0.056     |   0.162    |
| Word2Vec                      |   0.000     |     ???    |
| CodeBERT                      |   0.06      |     ???    |
| Leopard                       |   0.124     |   0.122    |
| Leopard + Naive (All)         |   0.129     |  0.1254    |
| Leopard + Naive (Top 20)      |   0.129     |  0.1253    |
| Leopard + Naive (Top 15)      |   0.137     |  0.1220    |
| Leopard + Naive (Top 10)      |   0.117     |  0.1362    |
| Hacking (All)                 |     ---     |   0.652    |


# Contributions

The approaches and strategies described in this project were discussed collaboratively in team meetings.
All implementation, paper reviewing, and literature searching were carried out by Lennart Dammer.





