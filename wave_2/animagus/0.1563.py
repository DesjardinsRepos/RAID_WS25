# Solution: encode the length of the encoded bytes in the first byte and the encoded bytes after (0.156)

def encode(carrier_text: bytes, to_encode: bytes) -> bytes:
    length_byte = len(to_encode).to_bytes(1, byteorder='big') # this wont work for larger to_encoded values
    encoded_part = length_byte + to_encode
    rest = carrier_text[len(encoded_part):]
    return encoded_part + rest

def decode(to_decode: bytes) -> bytes:
    if not to_decode:
        return b""

    length = to_decode[0]
    return to_decode[1:1+length]
