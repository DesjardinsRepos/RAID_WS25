import torch
import torch.nn.functional as F
from pathlib import Path
import pickle
import sys
import numpy as np

def infer_full(image_path, target):
    model = torch.load("final.pt", map_location=torch.device("cpu"), weights_only=False)
    model.eval()

    image = pickle.loads(Path(image_path).read_bytes())
    image = torch.from_numpy(image).unsqueeze(0).requires_grad_(True)

    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    pred_conf = probs[0, pred].item()
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()

    loss = F.cross_entropy(logits, torch.tensor([target]))
    loss.backward()
    grad = image.grad.squeeze(0).detach().numpy()

    result = {
        "target": target,
        "predicted": pred,
        "confidence": pred_conf,
        "entropy": entropy,
        "gradient": grad
    }

    Path("/tmp/grad-out.pkl").write_bytes(pickle.dumps(result))

if __name__ == "__main__":
    image_path = Path(sys.argv[1])
    target = int(sys.argv[sys.argv.index("--grad") + 1])
    infer_full(image_path, target)
