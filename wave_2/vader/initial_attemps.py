import numpy as np
import cv2

# Solution: calculate average pixel value (0.014)
def reconstruct(frames: list) -> np.ndarray:
    acc = np.zeros_like(frames[0], dtype=np.float32)
    for frame in frames:
        acc += frame.astype(np.float32)
    background_avg = acc / len(frames)
    background_img = np.clip(background_avg, 0, 255).astype(np.uint8)
    background_img = cv2.resize(background_img, (1280, 720))
    return background_img


# Solution: take the second most frequent pixel value (0.027)
def reconstruct(frames: list) -> np.ndarray:
    stack = np.stack(frames, axis=0)
    h, w, c = stack.shape[1:]
    out = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                vals = stack[:, i, j, k]
                uniq, counts = np.unique(vals, return_counts=True)
                if uniq.size > 1:
                    second = uniq[np.argsort(counts)[-2]]
                    out[i, j, k] = second
                else:
                    out[i, j, k] = uniq[0]
    out = cv2.resize(out, (1280, 720))
    return out