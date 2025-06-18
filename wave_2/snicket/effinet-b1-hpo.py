import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from tqdm import tqdm
from collections import Counter
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "snicket/data/train"
val_dir = "snicket/data/val"

data_transforms = {
    'all': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['all'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['all'])
trainval_dataset = ConcatDataset([train_dataset, val_dataset])

all_imgs = train_dataset.imgs + val_dataset.imgs
counts = Counter([label for _, label in all_imgs])
total = sum(counts.values())
class_weights_dict = {cls: total / count for cls, count in counts.items()}
sample_weights = [class_weights_dict[label] for _, label in all_imgs]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
trainval_loader = DataLoader(trainval_dataset, batch_size=32, sampler=sampler, num_workers=4)

def objective(trial):
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    wd = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.3)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    model.classifier[0] = nn.Dropout(p=dropout)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    for epoch in range(150):
        model.train()
        running_loss = 0
        correct, total = 0, 0
        for images, labels in trainval_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + images.size(0) / len(trainval_loader.dataset))

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        if epoch > 10:
            model_path = f"./models/trial{trial.number}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value:.4f}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
