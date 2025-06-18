def _bits(bs: bytes) -> str:
    """Convert bytes to a string of bits."""
    return ''.join(f'{b:08b}' for b in bs)

def _alpha_pos(buf: bytes):
    """Return positions of alphabetic characters in the buffer."""
    return [i for i,c in enumerate(buf) if 64 < c < 91 or 96 < c < 123]

def encode(carrier_text: bytes, to_encode: bytes) -> bytes:
    """
    Simple case modification encoder:
    - Uppercase = 1
    - Lowercase = 0
    No optimization, direct encoding of bits.
    """
    payload_len = len(to_encode)
    bits_to_encode = f'{payload_len:08b}' + _bits(to_encode)
    num_bits = len(bits_to_encode)
    
    # Get positions of alphabetic characters
    pos = _alpha_pos(carrier_text)
    
    # Check if we have enough capacity
    if len(pos) < num_bits:
        print(f"Error: carrier lacks capacity. Need {num_bits} alphabetic chars, have {len(pos)}")
        return carrier_text
        
    # Create mutable copy of carrier
    result = bytearray(carrier_text)
    
    # Encode each bit by setting case
    flip_count = 0
    for bit, idx in zip(bits_to_encode, pos):
        current_char = result[idx]
        is_upper = 64 < current_char < 91
        
        # If bit is '1', we want uppercase
        # If bit is '0', we want lowercase
        if (bit == '1') != is_upper:
            # Toggle case by XORing with 0x20
            result[idx] ^= 0x20
            flip_count += 1
            
    print("flips:", flip_count)
    return bytes(result) 