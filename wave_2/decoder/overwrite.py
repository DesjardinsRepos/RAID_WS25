def decode(to_decode: bytes) -> bytes:
    if not to_decode:
        return b""

    length = to_decode[0]
    return to_decode[1:1+length]