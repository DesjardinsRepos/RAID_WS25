import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import optuna

train_dir = "snicket/data/train"
val_dir = "snicket/data/val"

batch_size = 32
num_epochs = 10
learning_rate = 0.00017869207183459067
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

counts = Counter([label for _, label in train_dataset.imgs])
total = sum(counts.values())
class_weights = {cls: total / count for cls, count in counts.items()}
sample_weights = [class_weights[label] for _, label in train_dataset.imgs]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

def objective(trial):
    class_weight = trial.suggest_float("class_weight", 0.2, 5.0)

    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    model.classifier[0] = nn.Dropout(p=0.37622724439090177)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    class_counts = [counts[0], counts[1]]
    weights = torch.tensor([class_weight, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.13981704058385433)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.705406839352058e-05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step(train_loss)

        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total

        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch > 10:
            model_save_path = f"models/efficientnet_b1_trial{trial.number}_epoch{epoch}.pth"
            torch.save(model.state_dict(), model_save_path)

    return val_accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value:.4f}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
