from typing import List
import re
import numpy as np


def calc_features(functions: List[str]) -> List[List[float]]:
    # Define patterns for feature extraction
    patterns = {
        'integer_overflow': r'(\+\+|\+=|--|-=|\*=|\/=)',
        'input_validation': r'(if\s*\(\s*\w+\s*(==|!=|<|>|<=|>=))',
        'error_handling': r'(try|catch|throw|except|finally|raise)',
        'null_checks': r'(==\s*NULL|!=\s*NULL|==\s*nullptr|!=\s*nullptr)',
        'buffer_operations': r'\b(memcpy|strcpy|strcat|sprintf|gets|scanf)\b',
        'memory_alloc': r'\b(malloc|calloc|realloc|alloc|new\s+[\w\[\]<>]+)\b',
        'memory_free': r'\b(free|delete)\b'
    }
    
    features = []
    for func in functions:
        # Calculate function length
        func_length = len(func)
        
        # Initialize feature vector
        feature_vector = [func_length]
        
        # Count occurrences of each pattern
        for pattern in patterns.values():
            count = len(re.findall(pattern, func))
            feature_vector.append(count)
        
        features.append(feature_vector)
    
    # Convert to numpy array for scaling
    features_array = np.array(features)
    
    return features_array.tolist()
   