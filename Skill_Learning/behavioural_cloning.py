import os
import json
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from skill_helpers import *
from joblib import dump
import pandas as pd 
from math import isnan
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class BCDataset(Dataset):
    """
    Step-level tuples for MLP:
    Returns (state_t, action_t) where state_t is standardized float32, action_t is Long.
    """
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class PolicyMLP(nn.Module):
    def __init__(self, d_in, n_actions=17, hidden_sizes=(256, 128, 64), p_drop=0.1):
        super().__init__()
        layers = [nn.LayerNorm(d_in)]
        dim = d_in
        for h in hidden_sizes:
            layers += [nn.Linear(dim, h), nn.GELU(), nn.Dropout(p_drop)]
            dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, n_actions)

    def forward(self, x):
        return self.head(self.backbone(x))


dir_ = 'Craftax-Skill-Data/Traces/stone_pickaxe_easy'
files = os.listdir(os.path.join(dir_, 'groundTruth'))

unique_skills = get_unique_skills(dir_, files)
skill = "stone"
# for skill in unique_skills:

print(f"===== TRAINING SKILL {skill} =====")
episodes = get_bc_data_by_episode(dir_, files, skill, feature_name='pca_features_512')
rng = np.random.default_rng(0)
idx = np.arange(len(episodes))
rng.shuffle(idx)
n = len(idx)
train_idx = idx[: int(0.8*n)]
val_idx   = idx[int(0.8*n): int(0.9*n)]
test_idx  = idx[int(0.9*n):]

train_eps = [episodes[i] for i in train_idx]
val_eps   = [episodes[i] for i in val_idx]
test_eps  = [episodes[i] for i in test_idx]


X_tr, y_tr = bc_flatten_split(train_eps, use_skill=True)
X_va, y_va = bc_flatten_split(val_eps,   use_skill=True)
X_te, y_te = bc_flatten_split(test_eps,  use_skill=True)


print(X_tr.shape, y_tr.shape)
print(X_va.shape, y_va.shape)
print(X_te.shape, y_te.shape)


scaler = Standardizer()
X_tr = scaler.fit_transform(X_tr)
X_va = scaler.transform(X_va)
X_te = scaler.transform(X_te)


train_ds = BCDataset(X_tr, y_tr)
val_ds   = BCDataset(X_va, y_va)
test_ds  = BCDataset(X_te, y_te)

n_actions = 16
counts = np.bincount(y_tr, minlength=n_actions).astype(np.float64)
inv = np.zeros_like(counts); obs = counts > 0
inv[obs] = 1.0 / counts[obs]
sample_w = inv[y_tr]  # per-sample weight = inverse freq of its class
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

use_mps = torch.backends.mps.is_available()
train_loader = DataLoader(train_ds, batch_size=1024, sampler=sampler, shuffle=False, pin_memory=not use_mps) 
val_loader   = DataLoader(val_ds,   batch_size=2048, shuffle=False, pin_memory=not use_mps)
test_loader  = DataLoader(test_ds,  batch_size=2048, shuffle=False, pin_memory=not use_mps)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PolicyMLP(d_in=X_tr.shape[1], n_actions=n_actions, hidden_sizes=(384, 252)).to(device)

criterion = nn.CrossEntropyLoss()  # keep weights OFF
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

best_val = float('inf'); patience=10; bad=0
for epoch in range(168):
    # train
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * xb.size(0)
    train_loss = total / len(train_ds)

    # val
    model.eval()
    with torch.no_grad():
        tot, correct = 0.0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
        val_loss = tot / len(val_ds)
        val_acc = correct / len(val_ds)

    print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")

    # early stop on val CE
    if val_loss + 1e-6 < best_val:
        best_val = val_loss
        bad = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # else:
    #     bad += 1
    #     if bad >= patience:
    #         break


model.load_state_dict(best_state)
model.to(device)

# quick test metric
model.eval()
with torch.no_grad():
    tot, correct = 0.0, 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        tot += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
    print(f"TEST  NLL {tot/len(test_ds):.4f} | ACC {correct/len(test_ds):.3f}")

print("========================\n")



# #Save model and scaler to disk
# torch.save(model.state_dict(), f"model_{skill}.pt")
# joblib.dump(scaler, f"scaler_{skill}.pkl")
