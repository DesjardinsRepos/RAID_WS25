import os
os.environ["TORCH_HOME"] = "/home/user/.cache"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from pathlib import Path
from typing import List
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

def classify_images(
    img_paths: List[Path],
    model_path: str = "model.pth"
) -> List[bool]:
    device = "cpu"

    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    num_classes = 2
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((240, 240)),  # Input size for B1
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
