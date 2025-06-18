import os
os.environ["TORCH_HOME"] = "/home/user/.cache"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from pathlib import Path
from typing import List
from torchvision import models, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import torch
import torch.nn as nn


def classify_images(
    img_paths: List[Path],
    model_path: str = "model.pth"
) -> List[bool]:
    device = "cpu"

    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    num_classes = 2
    model.classifier[0] = nn.Dropout(p=0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    predictions: List[bool] = []

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        is_generated = (predicted_class == 0)
        predictions.append(is_generated)

    return predictions
