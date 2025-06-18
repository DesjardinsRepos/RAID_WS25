def encode(carrier_text: bytes, to_encode: bytes) -> bytes:
    """
    Encodes a message by placing it at the beginning of the file.
    First strips trailing spaces before newlines to optimize space usage.
    """
    assert len(to_encode) <= 255, "Payload too large"
    assert len(carrier_text) <= 1024, "Carrier too large"

    lines = carrier_text.split(b'\n')
    cleaned_lines = []
    freed_bytes = 0

    for line in lines[:-1]:
        stripped = line.rstrip(b' ')
        freed = len(line) - len(stripped)
        freed_bytes += freed
        cleaned_lines.append(stripped)
    cleaned_lines.append(lines[-1])  # Preserve final line

    cleaned_text = b'\n'.join(cleaned_lines)

    # Encode length (1 byte) + payload at the beginning
    length_byte = len(to_encode).to_bytes(1, 'big')
    payload_with_length = length_byte + to_encode
    
    # Calculate how much of the cleaned carrier text we can include
    remaining_space = len(carrier_text) - len(payload_with_length)
    if remaining_space < 0:
        raise ValueError("Carrier file too small for this message")
    
    # Take only the part of cleaned text that fits
    remaining_carrier = cleaned_text[:remaining_space]
    
    return payload_with_length + remaining_carrier