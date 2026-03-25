#Behavioral Cloning is supervised learning on expert demonstrations.
#trains ONLY the Transformer feature extractor.
#LSTM gets trained later during PPO.

import torch
import torch.nn as nn
import numpy as np
import os
import json
from transformer import TransformerFeatures
from gymnasium import spaces
class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        self.features = TransformerFeatures(
            observation_space=obs_space,
            features_dim=128,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_ff=128,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )
    def forward(self, x):
        features = self.features(x)
        return self.head(features)


def train_bc(data_dir="data/expert", epochs=30, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")
    states = np.load(os.path.join(data_dir, "states.npy"))
    actions = np.load(os.path.join(data_dir, "actions.npy"))
    print(f"loaded {len(states)} state and action pairs")
    #traintest split
    split = int(0.9 * len(states))
    train_x = torch.tensor(states[:split], dtype=torch.float32)
    train_y = torch.tensor(actions[:split], dtype=torch.long)
    val_x = torch.tensor(states[split:], dtype=torch.float32)
    val_y = torch.tensor(actions[split:], dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"Train: {len(train_x)} | Val: {len(val_x)}")
    model = BCPolicy().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"total no. of parameters: {total:,}")

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(   #half the learning rate every 10 epochs
        optimizer, step_size=10, gamma=0.5
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_acc = 0.0
    os.makedirs("models/bc", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += batch_y.size(0)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_total += batch_y.size(0)

        t_loss = train_loss / train_total
        t_acc = 100.0 * train_correct / train_total
        v_loss = val_loss / val_total
        v_acc = 100.0 * val_correct / val_total

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "features_state_dict": model.features.state_dict(),
                "val_acc": v_acc,
                "epoch": epoch,
            }, "models/bc/best_bc.pth")
            print(f"  [BEST] Saved best model (val acc: {v_acc:.1f}%)")

        scheduler.step()

    with open("models/bc/history.json", "w") as f:
        json.dump(history, f)

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print(f"Saved to: models/bc/best_bc.pth")
    print(f"History: models/bc/history.json")


if __name__ == "__main__":
    train_bc(epochs=30, batch_size=256, lr=1e-3)