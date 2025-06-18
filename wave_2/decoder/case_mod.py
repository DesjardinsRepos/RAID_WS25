def decode(to_decode: bytes) -> bytes:
    p = [i for i, c in enumerate(to_decode) if 64 < c < 91 or 96 < c < 123]
    l = int(''.join('1' if 64 < to_decode[i] < 91 else '0' for i in p[:8]), 2)
    b = ''.join('1' if 64 < to_decode[i] < 91 else '0' for i in p[8:8+l*8])
    return bytes(int(b[i:i+8], 2) for i in range(0, len(b), 8)) 