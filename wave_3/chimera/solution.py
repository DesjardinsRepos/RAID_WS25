import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import random
from pathlib import Path
import pickle
from subprocess import run

from util import is_chimera, torch_float32_to_np_uint8, np_uint8_to_torch_float32

# STE quantization to keep gradients flowing
class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_np = torch_float32_to_np_uint8(input)
        output = np_uint8_to_torch_float32(input_np)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize(tensor):
    return QuantizeSTE.apply(tensor)

def hash_image(tensor: torch.Tensor) -> str:
    img_bytes = torch_float32_to_np_uint8(tensor).tobytes()
    return hashlib.sha256(img_bytes).hexdigest()

def hash_raw(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

def get_gradient(backend, image_path, target):
    run([f"python-{backend}", "models-edited.py", image_path, "--grad", str(target)])
    result = pickle.loads(Path("/tmp/grad-out.pkl").read_bytes())

    print(
        "  \u2192 Target " + str(result["target"]) + " | Backend: " + backend + " | " +
        "Predicted: " + str(result["predicted"]) + ", " +
        "Confidence: {:.4f}".format(result["confidence"]) + ", Entropy: {:.4f}".format(result["entropy"]),
        flush=True,
    )

    return torch.tensor(result["gradient"]), result["predicted"]


def create_chimera(image: np.ndarray, label: int):
    device = torch.device("cpu")
    model = torch.load("final.pt", map_location=device, weights_only=False)
    model.eval()

    # Prepare initial image tensor and reparameterize
    image_torch = torch.tensor(image, dtype=torch.float32, device=device)
    image_clamped = (image_torch * 2 - 1).clamp(-0.999, 0.999)
    w_init = torch.atanh(image_clamped).detach()
    w = w_init.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=1e-2)

    # Step 1: Pull towards decision boundary

    lr = 1e-1
    max_steps = 200
    m = torch.zeros_like(w)
    v = torch.zeros_like(w)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-7

    prev_predicted = label  # Start with true label
    x_q_prev = None  # Previous quantized image for distance calculation
    pred_counts = {}

    for i in range(1, max_steps + 1):
        x = 0.5 * (torch.tanh(w) + 1)
        x_q = quantize(x)

        out = model(x_q.unsqueeze(0))
        probs = torch.softmax(out, dim=1)

        predicted = torch.argmax(probs, dim=1).item()
        predicted_conf = probs[0, predicted].item()
        pred_counts[predicted] = pred_counts.get(predicted, 0) + 1

        # If predicted class flipped, lower the step size
        if predicted != prev_predicted:
            lr *= 0.995
        prev_predicted = predicted

        loss = F.cross_entropy(out, torch.tensor([label], device=device))

        # Gradient ascent if correct, else descent
        grad_direction = 1.0 if predicted == label else -1.0

        optimizer.zero_grad()
        loss.backward()
        grad = w.grad.detach()

        # Manual adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad.pow(2)
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)

        update = grad_direction * lr * m_hat / (v_hat.sqrt() + eps)
        w.data += update

        # Metrics
        step_distance = 0.0
        if x_q_prev is not None:
            step_distance = (x_q - x_q_prev).norm().item()
        x_q_prev = x_q.detach()
        update_magnitude = update.norm().item()

        print(
            f"Step {i} - Δw: {step_distance:.6f}, step size: {update_magnitude:.6f}, "
            f"lr: {lr:.5e}, predicted: {predicted}, pred conf: {predicted_conf:.4f}",
            flush=True,
        )
        
    # Step 2: Chimera search
    pred_counts.pop(label, None)
    most_freq_label = max(pred_counts, key=pred_counts.get)
    targets = [label, most_freq_label]
    print(f"[INFO] Target classes for Chimera search: {targets}", flush=True)

    lr = 1e-2

    for i in range(3000):
        x = 0.5 * (torch.tanh(w) + 1)
        x_q = quantize(x)

        try:
            if is_chimera(x_q.cpu().detach().numpy()):
                print("🎯 Chimera found!", flush=True)
                return torch_float32_to_np_uint8(x_q)
        except Exception as e:
            print(f"[ERROR] is_chimera failed: {e}", flush=True)

        image_path = Path("/tmp/image.pkl")
        image_path.write_bytes(pickle.dumps(x_q.detach().numpy()))

        grad_accum = torch.zeros_like(w)

        grad_openblas, pred_openblas = get_gradient("openblas", image_path, targets[0])
        grad_blis, pred_blis = get_gradient("blis", image_path, targets[1])

        grad_accum = grad_openblas + grad_blis
        print(
            f"  → OpenBLAS Gradnorm: {grad_openblas.norm():.4f} | "
            f"BLIS Gradnorm: {grad_blis.norm():.4f}",
            flush=True,
        )

        grad_accum /= len(targets)
        w.data -= lr * grad_accum # FIXME should this be +?
        print(f"Step {i} - grad norm: {grad_accum.norm():.6f}", flush=True)

    print("❌ Chimera not found after max iterations.", flush=True)
    return x_q.cpu().detach().numpy()