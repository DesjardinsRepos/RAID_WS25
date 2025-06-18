def decode(to_decode: bytes) -> bytes:
    length_byte = to_decode[:1]
    message_length = int.from_bytes(length_byte, 'big')
    
    return to_decode[1:1 + message_length]