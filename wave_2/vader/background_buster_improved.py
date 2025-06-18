import os
os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.cache")

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import traceback

import numpy as np
import cv2
from collections import Counter
from pathlib import Path
from scipy import stats

import torch
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet101


# SCORE: 0.0615 mit temporal_majority_vcm

deeplab_model = None

try:
    deeplab_model = deeplabv3_resnet101(weights=None, weights_backbone=None, aux_loss=True)
except Exception as e:
    print("Fehler beim Laden des Modells:", e)
    traceback.print_exc()

if deeplab_model is None:
    raise RuntimeError("Modell konnte nicht initialisiert werden.")

deeplab_model.load_state_dict(torch.load("deeplabv3_resnet101_coco.pt", map_location="cpu"))
deeplab_model.eval()
device = torch.device("cpu")
deeplab_model.to(device)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def reconstruct(frames: list) -> np.ndarray:
    # Hintergrund über Frames mitteln
    vb_img = estimate_vb(frames)

    print("Virtual background: Done")


    leaked_frames = []
    for i, frame in enumerate(frames):
        vbm = compute_virtual_background_mask_adaptive(frame, vb_img, 0.05)
        #bbm = compute_blend_mask(vbm, radius=5)
        vcm = extract_person_mask_deeplab(frame)

    
        #leak_mask = (1 - vbm) * (1 - vcm) * (1 - bbm)
        #leak_mask = ((vbm == 0) & (vcm == 0) & (bbm == 0)).astype(np.uint8)
        leak_mask = ((vbm == 0) & (vcm == 0)).astype(np.uint8)

        leak_rgb = frame * leak_mask[:, :, np.newaxis]
        leaked_frames.append(leak_rgb)



    print("Leaked frames processed and saved.")
    leak_imgs = np.stack(leaked_frames)
    estimated_bg = reconstruct_background_per_pixel(leak_imgs, vb_img)

    black_mask = np.all(estimated_bg == 0, axis=-1).astype(np.uint8)
    inpainted_telea = cv2.inpaint(estimated_bg, black_mask, 3, cv2.INPAINT_TELEA)

    return inpainted_telea



def estimate_vb(frames: list[np.ndarray], threshold: int = 10) -> np.ndarray:
    video_array = np.stack(frames, axis=0)  # Shape: (T, H, W, C)
    T, H, W, C = video_array.shape

    # Wir reshapen zu (T, H*W, 3) und machen dann Trick mit RGB als Tuple
    reshaped = video_array.reshape(T, -1, 3)  # (T, H*W, 3)

    # Konvertiere jede RGB-Farbe in einen Integer-Wert
    rgb_flat = reshaped[:, :, 0].astype(np.uint32) << 16 | \
               reshaped[:, :, 1].astype(np.uint32) << 8 | \
               reshaped[:, :, 2].astype(np.uint32)

    # Modus (häufigster Wert) entlang der Zeitachse
    mode, count = stats.mode(rgb_flat, axis=0, keepdims=False)

    # Nur behalten, wenn dieser Wert mindestens `threshold` mal vorkommt
    vb_flat = np.zeros_like(mode, dtype=np.uint32)
    vb_flat[count >= threshold] = mode[count >= threshold]

    # Rückkonvertierung in RGB
    r = (vb_flat >> 16) & 0xFF
    g = (vb_flat >> 8) & 0xFF
    b = vb_flat & 0xFF
    vb_rgb = np.stack([r, g, b], axis=1).astype(np.uint8)

    # Zurück zu Bildform
    vb_img = vb_rgb.reshape(H, W, 3)
    return vb_img

def compute_virtual_background_mask_adaptive(frame: np.ndarray, vb_img: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Erzeugt adaptiv eine Maske, die zeigt, welche Pixel zum virtuellen Hintergrund gehören.
    Farbdifferenz zwischen Frame und geschätztem VB wird ausgewertet.
    """
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
    vb_lab = cv2.cvtColor(vb_img, cv2.COLOR_BGR2Lab).astype(np.float32)

    delta = np.linalg.norm(frame_lab - vb_lab, axis=2)  # L2-Abstand

    mean = np.mean(delta)
    std = np.std(delta)
    thresh = mean + alpha * std

    vbm = (delta < thresh).astype(np.uint8)

    # Optional glätten
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vbm = cv2.morphologyEx(vbm, cv2.MORPH_CLOSE, kernel)

    return vbm

def compute_blend_mask(vbm: np.ndarray, radius: int) -> np.ndarray:
    # 0–1 Float → 0–255 uint8
    vbm_u8 = (vbm * 255).astype(np.uint8)
    # binarisieren bei 127 (also 0.5)
    _, vbm_bin = cv2.threshold(vbm_u8, 127, 1, cv2.THRESH_BINARY)

    k = 2*radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(vbm_bin, kernel)
    # BBM = dilated − original
    bbm = cv2.subtract(dilated, vbm_bin)
    return bbm

def extract_person_mask_deeplab(frame: np.ndarray) -> np.ndarray:
    """
    Gibt eine Binärmaske (1 = Person) zurück für ein einzelnes Frame.
    """
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab_model(input_tensor)['out'][0]
    predictions = torch.argmax(output, dim=0).cpu().numpy()
    person_mask = (predictions == 15).astype(np.uint8)  # Klasse 15 = 'person'
    person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return person_mask

def temporal_majority_vcm(vcm_list: list[np.ndarray], window_size: int = 3) -> list[np.ndarray]:
    """
    Stabilisierte Person-Maske durch Mehrheitsentscheidung (Voting) über window_size Frames.
    """
    assert window_size % 2 == 1, "Window size muss ungerade sein"
    half = window_size // 2
    num_frames = len(vcm_list)
    smoothed_masks = []

    for i in range(num_frames):
        start = max(0, i - half)
        end = min(num_frames, i + half + 1)
        window = np.stack(vcm_list[start:end], axis=0)
        majority = (np.sum(window, axis=0) > (window.shape[0] // 2)).astype(np.uint8)
        smoothed_masks.append(majority)

    return smoothed_masks


def reconstruct_background_per_pixel(leak_imgs: np.ndarray, vb_image: np.ndarray) -> np.ndarray:
    """
    leak_imgs: numpy array of shape (N, H, W, 3), dtype float32 or uint8
    vb_image: numpy array of shape (H, W, 3), dtype float32 or uint8
    """
    if leak_imgs.dtype != np.float32:
        leak_imgs = leak_imgs.astype(np.float32)
    if vb_image.dtype != np.float32:
        vb_image = vb_image.astype(np.float32)
    N, H, W, C = leak_imgs.shape

    # Compute per-pixel L2 distance to virtual background
    diff = np.linalg.norm(leak_imgs - vb_image, axis=3)  # (N, H, W)

    # Mask out invalid (black) pixels in leak images
    valid = np.any(leak_imgs != 0, axis=3)  # (N, H, W)
    diff[~valid] = -np.inf  # So they are never selected

    # For each pixel, select the leak image with the largest difference from vb (most likely real background)
    best_img_idx = np.argmax(diff, axis=0)  # (H, W)
    best_diff = np.max(diff, axis=0)  # (H, W)

    # Build the reconstructed image
    reconstructed = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            idx = best_img_idx[i, j]
            if best_diff[i, j] > 0:  # Only fill if valid
                reconstructed[i, j] = leak_imgs[idx, i, j]
            else:
                reconstructed[i, j] = 0  # set to black if no valid leak

    return reconstructed.astype(np.uint8)


# from util import load_video, load_background, load_mask, metric_CIEDE2000, evaluate

# if __name__ == "__main__":
#     frames = load_video(Path("data/public/videos/2_i_kitchen_leaves_mp.mp4"))
#     reconstruct(frames)