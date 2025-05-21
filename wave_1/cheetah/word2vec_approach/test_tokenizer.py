import user.word2vec_approach.code_tokenizer as code_tokenizer
from solution import calc_features

def test_tokenization():
    """Test the tokenization functionality on a simple C function."""
    test_function = """
    unsigned WebGraphicsContext3DDefaultImpl::createBuffer()\n{\n    makeContextCurrent();\n    GLuint o;\n    glGenBuffersARB(1, &o);\n    return o;\n}\n
    """
    
    # Test tokenize_code function
    tokens = code_tokenizer.tokenize_code(test_function)
    print(f"Tokenized function ({len(tokens)} tokens):")
    print(tokens)
    
  
    # Test calc_features function
    combined_features = calc_features([test_function])
    print("\nCombined features (traditional + tokenization):")
    print(combined_features[0])
    print(f"Feature vector length: {len(combined_features[0])}")

if __name__ == "__main__":
    test_tokenization() 