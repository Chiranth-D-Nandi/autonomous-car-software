#mobile net v3 small training code
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import json
from datetime import datetime
import wandb

class Classifier(nn.Module):
    CLASSES = ["green", "red", "stop_sign"]
    NUM_CLASSES = 3
    def __init__(self, pretrained=True, dropout = 0.3):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                                             if pretrained else None)
        for i, (name, param) in enumerate(backbone.features.named_parameters()):
            if i < 40:
                param.requires_grad=False
        self.features = backbone.features #layer that extracts features from img
        self.pool = nn.AdaptiveAvgPool2d((1,1)) #combines those features to a tensor, didn't use avgpool to accomodate varied img input sizes
        self.classifier = nn.Sequential(
            nn.Linear(576, 128),       
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(128, self.NUM_CLASSES)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def data_processing():
    train_transform = transforms.Compose([transforms.Resize((96,96)), transforms.RandomHorizontalFlip(p=0.4),transforms.RandomRotation(20),transforms.RandomPerspective(distortion_scale=0.3, p = 0.3), transforms.ColorJitter(brightness=0.3, contrast = 0.3, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_transform, val_transform

def train_classifier(data_dir='data/traffic_signs', epochs=50, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")
    #load the data
    train_transform, val_transform = data_processing()
    train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform = train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, "val"), transform = val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"no. of training imgs: {len(train_dataset)}, val: {len(val_dataset)}")
    #model / forwarding
    model = Classifier(pretrained=True, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params={trainable_params} out of {total_params}.")
    #loss and back propagation, optimisation
    criterion = nn.CrossEntropyLoss(label_smoothing =0.1) #crossentropyloss is standard for classification, binarycrossentropyloss for binary class.
    #label smoothing has more decimals in the tensor
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=1e-4)
    scheduler= CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) #smoother than step decay(imp)

    wandb.init(project="traffic signs classifier", mode="disabled", config={
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "architecture": "MobileNetV3-Small", "loss": "CrossEntropy+LabelSmoothing",})
    
    #training loop
    best_val_acc = 0.0 #start w/ accuracy 0
    patience = 10 #stop if acc doesnt change after 10 iterations
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0) #avg loss per img * number of imgs in batch
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        per_class_correct = [0] * Classifier.NUM_CLASSES
        per_class_total = [0] * Classifier.NUM_CLASSES
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    per_class_total[label] += 1
                    if predicted[i].item() == label:
                        per_class_correct[label] += 1
        
        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        log_dict = {
            "train/loss": train_loss, "train/accuracy": train_acc,
            "val/loss": val_loss, "val/accuracy": val_acc,
            "lr": current_lr, "epoch": epoch,
        }
        for c in range(Classifier.NUM_CLASSES):
            if per_class_total[c] > 0:
                log_dict[f"val/acc_{Classifier.CLASSES[c]}"] = (
                    100.0 * per_class_correct[c] / per_class_total[c]
                )
        wandb.log(log_dict)
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f}")
        
        for c in range(Classifier.NUM_CLASSES):
            if per_class_total[c] > 0:
                class_acc = 100.0 * per_class_correct[c] / per_class_total[c]
                print(f"  {Classifier.CLASSES[c]}: {class_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': Classifier.CLASSES,
            }, "models/classifier/best_classifier.pth")
            print(f"new best model found. accuracy score: {val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    with open("models/classifier/training_history.json", "w") as f:
        json.dump(history, f)
    
    if True:
        wandb.finish()  
    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    return model, history


if __name__ == "__main__":
    os.makedirs("models/classifier", exist_ok=True)
    model, history = train_classifier(epochs=50, batch_size=128, lr=1e-3)
