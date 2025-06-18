def encode(carrier_text: bytes, to_encode: bytes) -> bytes:
    length_byte = len(to_encode).to_bytes(1, byteorder='big') # this wont work for larger to_encoded values
    encoded_part = length_byte + to_encode
    rest = carrier_text[len(encoded_part):]
    return encoded_part + rest

