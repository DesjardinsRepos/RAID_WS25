# ============================================================
# Systematic stego  (length 16–32 B)   –  GREEDY, 4-ones P
#     * first 5 letters → payload length code (0-16 ⇒ 16-32 B)
#     * G = [ I_k | P ] ,  k = 8·len(secret)  (128…256 bits)
#     * each parity column toggles 4 rows  (one deterministic + 3 PRNG)
# ============================================================
_MASK      = 0x20
LEN_BITS   = 5
MIN_BYTES  = 16
MAX_BYTES  = 32

# ---------- helpers ----------
def _alpha_pos(buf: bytes):
    return [i for i,c in enumerate(buf) if 64 < c < 91 or 96 < c < 123]

def _bits(bs: bytes) -> str:
    return ''.join(f'{b:08b}' for b in bs)

def _ibits(x:int,w:int)->str: return format(x,f'0{w}b')

def _col_rows(j:int,k:int):
    """return rows toggled by column j (tuple of 1–4 ints)"""
    if j < k: return (j,)
    r0 = (j - k) % k
    x  = (0x9E3779B97F4A7C15 * j + 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF
    r1 = (x       & 0xFF) % k
    r2 = ((x>> 8) & 0xFF) % k
    r3 = ((x>>16) & 0xFF) % k
    return tuple({r0,r1,r2,r3})         # dedup → 1-4 rows

# ---------- encoder ----------
def encode(carrier_text: bytes, to_encode: bytes) -> bytes:
    print(f"Encoding {len(to_encode)} bytes")
    carrier = carrier_text  # carrier_text is already bytes
    nB = len(to_encode)
    if not MIN_BYTES <= nB <= MAX_BYTES:
        print(f"Error: payload size {nB} bytes not in valid range {MIN_BYTES}-{MAX_BYTES}")
        return carrier
    pos = _alpha_pos(carrier)
    k   = nB*8
    need = LEN_BITS + k
    if len(pos) < need:
        print("Error: carrier lacks alphabetic capacity")
        return carrier

    ba = bytearray(carrier)

    # (1) store length code
    for bit,idx in zip(_ibits(nB-MIN_BYTES, LEN_BITS), pos[:LEN_BITS]):
        if (bit=='1') != (64 < ba[idx] < 91):
            ba[idx] ^= _MASK

    # (2) payload
    data_pos = pos[LEN_BITS:]
    n_cols   = len(data_pos)
    v        = [64 < ba[p] < 91 for p in data_pos]

    # current syndrome δ
    row_sum = [0]*k
    for j in range(k, n_cols):
        for r in _col_rows(j,k): row_sum[r] ^= v[j]
    delta = {i for i,mbit in enumerate(_bits(to_encode))
             if int(mbit) ^ v[i] ^ row_sum[i]}

    flips = []

    # Greedy pass over parity columns: keep only if it shrinks |δ|
    for j in range(k, n_cols):
        rows = _col_rows(j,k)
        new_delta = delta.symmetric_difference(rows)  # toggle these rows
        if len(new_delta) < len(delta):               # improvement?
            delta = new_delta
            flips.append(j)
            if not delta:
                break

    # Remaining rows: flip identity columns (one per row)
    flips.extend(delta)

    flip_count = 0

    for col in flips:
        ba[data_pos[col]] ^= _MASK
        flip_count += 1
    print("flips:", flip_count)
    return bytes(ba)