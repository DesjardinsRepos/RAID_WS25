import numpy as np

def reconstruct1(frames: list) -> np.ndarray:
    """
    Rekonstruiert den echten Hintergrund durch Analyse von Leaks über mehrere Frames,
    ohne Nutzung von Ground-Truth-Masken oder bekannten virtuellen Hintergründen.
    """

    if not frames:
        raise ValueError("Leere Frame-Liste erhalten – überprüfe Input.")

    # In NumPy-Array konvertieren
    video = np.stack(frames).astype(np.float32)  # shape: (N, 720, 1280, 3)

    # Schritt 1: Schätze den virtuellen Hintergrund (per Median über Frames)
    virtual_bg = np.median(video, axis=0)

    # Schritt 2: Finde pro Frame die Pixel, die deutlich vom virtuellen Hintergrund abweichen
    diffs = np.linalg.norm(video - virtual_bg[None, ...], axis=-1)  # shape: (N, 720, 1280)
    leak_threshold = 30  # RGB-Distanzschwelle
    leak_mask = diffs > leak_threshold

    # Schritt 3: Nur "geleakte" Pixel in Rekonstruktion aufnehmen
    accumulated = np.zeros_like(virtual_bg, dtype=np.float32)
    count = np.zeros(virtual_bg.shape[:2], dtype=np.float32)

    for i in range(video.shape[0]):
        mask = leak_mask[i]
        accumulated[mask] += video[i][mask]
        count[mask] += 1

    # Schritt 4: Mittelwert bilden, wo Pixel oft genug vorkamen
    count = np.clip(count, 1, None)
    reconstructed = accumulated / count[..., None]

    return reconstructed.astype(np.uint8)