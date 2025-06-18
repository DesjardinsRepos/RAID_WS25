from util import load_video, load_background, load_mask, metric_CIEDE2000, evaluate
from pathlib import Path
import matplotlib.pyplot as plt
from storage import reconstruct1
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import cv2
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt

def reconstruct(frames: list) -> np.ndarray:
    # Hintergrund über Frames mitteln
    virtual_bg = estimate_vb_by_stability(frames)
    # virtual_bg = estimate_vb_by_mean(frames)
    # virtual_bg = frames[0].astype(np.float32)

    vbm_masks = compute_virtual_background_masks(frames, virtual_bg, threshold=10)

    bbm_masks = compute_blending_zone_masks(vbm_masks)

    vcm_masks = compute_variance_person_masks(frames, threshold=500)
    # vcm_masks = compute_deeplab_person_masks(frames)

    reconstructed = reconstruct_from_leaks(frames, vbm_masks, bbm_masks, vcm_masks, virtual_bg, "median")

    return reconstructed


def estimate_vb_by_mean(frames: list):
    mean = np.zeros_like(frames[0], dtype=np.float32)
    count = 0
    for f in frames:
        count += 1
        mean += (f.astype(np.float32) - mean) / count
    virtual_bg = mean
    return virtual_bg

def estimate_vb_by_stability(frames: list,
                             threshold: int = 10,
                             min_stable: int = 10) -> np.ndarray:
    """
    Schätzt den virtuellen Hintergrund per Pixel-Stabilität relativ zum
    MEDIAN aller Frames. Optional leichtes Blurring gegen Rauschen.
    """
    # → (N, H, W, 3)   int16 vermeidet Overflow
    video = np.stack([cv2.GaussianBlur(f, (5, 5), 0)
                      for f in frames]).astype(np.int16)

    median_img = np.median(video, axis=0)                 # robust gegen Ausreißer
    diff = np.abs(video - median_img)                     # (N,H,W,3)
    stable = np.all(diff <= threshold, axis=-1)           # (N,H,W) bool

    # Counts & Sum über stabile Frames
    stable_counts = stable.sum(axis=0).astype(np.uint16)  # (H,W)
    stable_sum    = (video * stable[..., None]).sum(axis=0)  # (H,W,3)

    # Falls genügend stabile Samples: Mittelwert, sonst Median
    background = median_img.astype(np.uint8)
    mask_valid = stable_counts >= min_stable
    background[mask_valid] = (stable_sum[mask_valid] /
                              stable_counts[mask_valid, None]).astype(np.uint8)

    return background


def compute_virtual_background_masks(frames: list,
                                     virtual_bg: np.ndarray,
                                     min_thresh: float = 6.0,
                                     max_thresh: float = 18.0) -> list:
    """
    ΔE-Verteilung im Frame → Otsu teilt in 'ähnlich' vs. 'anders'.
    clamp(min_thresh, max_thresh) verhindert Ausreißer.
    """
    vb_lab = rgb2lab(virtual_bg.astype(np.float32) / 255.0)
    masks = []

    for f in frames:
        f_lab   = rgb2lab(f.astype(np.float32) / 255.0)
        delta_e = np.linalg.norm(f_lab - vb_lab, axis=-1).astype(np.float32)

        # Histogram-Otsu für automatische Schwelle
        t_otsu = threshold_otsu(delta_e)
        t      = float(np.clip(t_otsu, min_thresh, max_thresh))

        masks.append(delta_e < t)

    return masks



def compute_blending_zone_masks(vbm_masks: list,
                                scale: float = 0.03,
                                min_r: int = 4,
                                max_r: int = 25) -> list:
    """
    Radius = scale * Bildhöhe  (≈3 %)   → clamp.
    """
    h = vbm_masks[0].shape[0]
    radius = int(np.clip(scale * h, min_r, max_r))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,
                                                           2 * radius + 1))
    bbm_masks = []
    for vbm in vbm_masks:
        dilated = cv2.dilate(vbm.astype(np.uint8) * 255, kernel) > 0
        bbm_masks.append(np.logical_and(dilated, ~vbm))

    return bbm_masks


def compute_variance_person_masks(frames: list,
                                threshold: float = 300.0,
                                closing_k: int = 7,
                                erode_k: int = 3,
                                adaptive: bool = True) -> list:
    video = np.stack(frames).astype(np.float32)
    median_img = np.median(video, axis=0)
    
    # Bewegungsschätzung zwischen aufeinanderfolgenden Frames
    # Wichtig: flow_magnitude als float32 initialisieren
    flow_magnitude = np.zeros_like(frames[0][..., 0], dtype=np.float32)
    
    for i in range(len(frames)-1):
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Berechne die Magnitude des Flows
        current_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitude += current_magnitude
    
    # Normalisiere die Flow-Magnitude
    flow_magnitude /= len(frames)-1

    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_k, closing_k))
    se_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))

    masks = []
    for f in frames:
        diff = np.abs(f.astype(np.float32) - median_img).sum(axis=-1)
        
        if adaptive:
            # Kombiniere Farbdifferenz mit Bewegung
            local_threshold = threshold * (1 + flow_magnitude / flow_magnitude.max())
            mask = diff > local_threshold
        else:
            mask = diff > threshold
            
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se_close)
        mask = cv2.erode(mask, se_erode, iterations=1)
        masks.append(mask.astype(bool))

    return masks




def compute_deeplab_person_masks(frames: list) -> list:
    """
    Nutzt DeepLabv3, um Personensegmentierungen (VCMᵢ) zu erzeugen.
    Gibt eine Maske pro Frame zurück (True = Pixel gehört zur Person).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(pretrained=True).to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((frames[0].shape[0], frames[0].shape[1])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    masks = []
    with torch.no_grad():
        for f in frames:
            input_tensor = transform(f).unsqueeze(0).to(device)
            output = model(input_tensor)["out"][0]
            pred = torch.argmax(output, dim=0).cpu().numpy()

            person_mask = pred == 15  # Klasse 15 = Person in COCO
            masks.append(person_mask)

    return masks

def reconstruct_from_leaks(frames,
                           vbm_masks,
                           bbm_masks,
                           vcm_masks,
                           min_leaks: int = 2,
                           fuse: str = "median",
                           fill_color=(0, 0, 0)):
    """
    Rekonstruiert den echten Hintergrund.
    Pixel ohne genügend Real-Leaks werden mit 'fill_color' gefüllt.
    """
    h, w, _ = frames[0].shape
    accum   = np.zeros((h, w, 3), dtype=np.float32)
    count   = np.zeros((h, w),     dtype=np.uint16)

    # ---------- Leaks aufsummieren ----------
    for i, frame in enumerate(frames):
        lbm = ~(vbm_masks[i] | bbm_masks[i] | vcm_masks[i])
        accum[lbm] += frame[lbm]
        count[lbm] += 1

    # Maske: mindestens 'min_leaks' echte Frames gesehen
    valid = count >= min_leaks

    # ---------- Ergebnis-Array anlegen ----------
    recon = np.zeros((h, w, 3), dtype=np.float32)
    recon[:] = fill_color                        # schwarz oder weiß

    if fuse == "median":
        # --- median nur über die gültigen Pixel berechnen ---
        stack = np.stack(frames).astype(np.float32)          # (N,H,W,3)
        iy, ix = np.where(valid)                             # Pixel-Indices
        for c in range(3):                                   # R,G,B separat
            recon[iy, ix, c] = np.median(
                stack[:, iy, ix, c], axis=0)
    else:  # "mean"
        recon[valid] = accum[valid] / count[valid, None]

    return recon.astype(np.uint8)




def _show(mask, title="Maske"):
    plt.figure(figsize=(4,4))
    plt.imshow(mask, cmap='gray')
    plt.title(title); plt.axis("off")

def _heatmap(arr, title="Count-Map"):
    plt.figure(figsize=(5,4))
    plt.imshow(arr, cmap='hot'); plt.colorbar(); plt.title(title); plt.axis("off")

def testing(verbose=True):
    frames = load_video(Path("data/public/videos/2_i_kitchen_leaves_mp.mp4"))
    # frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    vb     = estimate_vb_by_stability(frames)

    # ---------- adaptives Tuning -------------
    vbm = compute_virtual_background_masks(frames, vb, min_thresh=7, max_thresh=7)

    bbm = compute_blending_zone_masks(vbm, scale=0.02)
    bbm = [cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,(3,3))).astype(bool) for m in bbm]
    
    vcm = compute_variance_person_masks(frames, threshold=220)

    recon = reconstruct_from_leaks(frames, vbm, bbm, vcm,
                               min_leaks=2,
                               fuse="median",
                               fill_color=(0, 0, 0))


    gt_bg   = load_background(Path("data/public/backgrounds/kitchen.png"))
    gt_mask = load_mask(Path("data/public/masks/2_i_kitchen_mp.png"))
    dE      = metric_CIEDE2000(recon, gt_bg, gt_mask)
    score   = evaluate(dE, gt_mask)

    if verbose:
        for name, m in [("VBM", vbm[0]), ("BBM", bbm[0]),
                        ("VCM", vcm[0])]:
            plt.figure(figsize=(3,3)); plt.imshow(m, cmap="gray")
            plt.title(f"{name} (Frame 0)"); plt.axis("off")
        plt.figure(figsize=(5,4)); plt.imshow(recon)
        plt.title("Rekonstruierter Background"); plt.axis("off"); plt.show()

    print(f"Adaptive Reconstruction-Score: {score:.4f}")



if __name__ == '__main__':
    testing()