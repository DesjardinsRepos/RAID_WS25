import os
os.environ["TORCH_HOME"] = "/home/user/.cache"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from pathlib import Path
from typing import List
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.fft

def add_frequency_channel(img_tensor: torch.Tensor) -> torch.Tensor:
    gray = img_tensor.mean(dim=0, keepdim=True)
    fft = torch.fft.fft2(gray)
    fft_mag = torch.abs(fft)
    fft_log = torch.log(fft_mag + 1e-8)
    fft_norm = (fft_log - fft_log.min()) / (fft_log.max() - fft_log.min())
    return torch.cat([img_tensor, fft_norm], dim=0)

def classify_images(
    img_paths: List[Path],
    model_path: str = "model.pth"
) -> List[bool]:
    device = "cpu"

    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    orig_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=4,
        out_channels=orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=orig_conv.bias is not None
    )
    with torch.no_grad():
        model.features[0][0].weight[:, :3] = orig_conv.weight
        model.features[0][0].weight[:, 3:] = orig_conv.weight[:, :1]

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    predictions: List[bool] = []

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        rgb_tensor = preprocess(image)
        input_tensor = add_frequency_channel(rgb_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        is_generated = (predicted_class == 0)
        predictions.append(is_generated)

    return predictions
