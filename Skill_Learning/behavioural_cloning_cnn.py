import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import argparse
# NEW: optuna
import optuna

from skill_helpers import *

# --- NEW: load images per-episode (reuses your skill mask & actions logic) ---
def get_bc_images_by_episode(dir_, files, skill, image_dir_name='pixel_obs'):
    """
    Like get_bc_data_by_episode, but loads raw images instead of PCA features.
    Expects per-episode npy shaped (T, H, W, 3), float32 in [0,1].
    """
    episodes = []
    for file in files:
        with open(os.path.join(dir_, 'groundTruth', file), 'r') as f:
            lines = f.read().splitlines()  # len = T

        img_path = os.path.join(dir_, image_dir_name, file + '.npy')
        act_path = os.path.join(dir_, 'actions', file + '.npy')

        images  = np.load(img_path)   # [T, H, W, 3] float32
        actions = np.load(act_path)   # [T]

        if len(lines) != len(images) or len(images) != len(actions):
            raise ValueError(
                f"Length mismatch in {file}: "
                f"labels={len(lines)} images={len(images)} actions={len(actions)}"
            )

        skill_mask = np.array([lab == skill for lab in lines], dtype=bool)
        other_mask = ~skill_mask

        ep = dict(
            episode_id=file,
            # keep the same interface names as before, but these are images now
            skill_states=images[skill_mask],     # shape [Ns, H, W, 3]
            skill_actions=actions[skill_mask],
            other_states=images[other_mask],
            other_actions=actions[other_mask],
            images=images,
            actions=actions,
            skill_mask=skill_mask
        )
        episodes.append(ep)
    return episodes


def compute_channel_mean_std(X):
    """
    X: numpy array [N, H, W, 3], float32 in [0,1]
    Returns per-channel mean/std as tuples of floats.
    """
    N, H, W, C = X.shape
    n_pixels = N * H * W
    flat = X.reshape(-1, C).astype(np.float64)  # higher precision for sums
    chan_sum = flat.sum(axis=0)                             # [3]
    chan_sqsum = np.square(flat).sum(axis=0)                # [3]
    mean = chan_sum / n_pixels
    var = chan_sqsum / n_pixels - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())

# --- replaces Standardizer for images: just channel-wise normalize ---
class ImageNormalizer:
    """
    Normalizes CHW images using dataset-specific per-channel mean/std.
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
        # avoid tiny std that can explode activations
        self.std = torch.clamp(self.std, min=1e-3)

    def __call__(self, x):
        # x: [3,H,W] in [0,1]
        return (x - self.mean) / self.std

class ImageBCDataset(Dataset):
    """
    Frame-level tuples: returns (img_t, action_t)
    - img_t: torch.float32 [3, H, W], normalized
    - action_t: torch.long
    """
    def __init__(self, X, y, normalizer=None, augment=False):
        assert X.shape[0] == y.shape[0]
        self.X = X      # numpy: [N, H, W, 3]
        self.y = y      # numpy: [N]
        self.norm = normalizer if normalizer is not None else ImageNormalizer()
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def _to_chw(self, img):
        # NHWC -> CHW
        return torch.from_numpy(np.transpose(img, (2,0,1))).float()

    # replace _random_crop_or_resize with resize-only
    def _resize(self, x, target=256):
        x = x.unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
        return x.squeeze(0)

    def __getitem__(self, idx):
        img = self._to_chw(self.X[idx])  # [3,H,W] in [0,1]
        img = self._resize(img, target=256)   # keep entire board
        img = self.norm(img)
        y = torch.tensor(self.y[idx]).long()
        return img, y


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

class PolicyCNN(nn.Module):
    def __init__(self, n_actions=16, width_mult=1.0, drop=0.0):
        super().__init__()
        w = lambda c: max(8, int(c * width_mult))
        # 256x256 -> downsample a few times
        self.stem = nn.Sequential(
            ConvBlock(3,   w(32), k=7, s=2, p=3, drop=drop),   # 128x128
            ConvBlock(w(32),  w(32), drop=drop),
            nn.MaxPool2d(2),                                  # 64x64
        )
        self.stage2 = nn.Sequential(
            ConvBlock(w(32),  w(64), drop=drop),
            ConvBlock(w(64),  w(64), drop=drop),
            nn.MaxPool2d(2),                                  # 32x32
        )
        self.stage3 = nn.Sequential(
            ConvBlock(w(64),  w(128), drop=drop),
            ConvBlock(w(128), w(128), drop=drop),
            nn.MaxPool2d(2),                                  # 16x16
        )
        self.stage4 = nn.Sequential(
            ConvBlock(w(128), w(256), drop=drop),
            ConvBlock(w(256), w(256), drop=drop),
        )
        self.head = nn.Linear(w(256), n_actions)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)   # [B, C, 1, 1]
        x = torch.flatten(x, 1)           # [B, C]
        return self.head(x)

# -------------- Args --------------
parser = argparse.ArgumentParser(description="Train CNN policy for a specific skill (with optional Optuna study)")
parser.add_argument("--skill", type=str, default="wood", help="Skill to train.")
# NEW: study knobs
parser.add_argument("--study", action="store_true", help="Run Optuna hyperparameter search instead of a single training run.")
parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials.")
parser.add_argument("--prune", action="store_true", help="Enable Optuna pruning.")
parser.add_argument("--epochs", type=int, default=60, help="Max epochs per (single run or trial).")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
args = parser.parse_args()

# -------------- Data prep (shared) --------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)

dir_ = '../Craftax/Traces/stone_pickaxe_easy'
files = os.listdir(os.path.join(dir_, 'groundTruth'))

unique_skills = get_unique_skills(dir_, files)
skill = args.skill

print(f"===== TRAINING SKILL {skill} (CNN) =====")
episodes = get_bc_images_by_episode(dir_, files, skill, image_dir_name='pixel_obs')

# checkpoint directory/path
ckpt_dir = os.path.join(dir_, 'bc_checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
default_ckpt_path = os.path.join(ckpt_dir, f'{skill}_policy_cnn.pt')

# episode split
rng = np.random.default_rng(args.seed)
idx = np.arange(len(episodes))
rng.shuffle(idx)
n = len(idx)
train_idx = idx[: int(0.8*n)]
val_idx   = idx[int(0.8*n): int(0.9*n)]
test_idx  = idx[int(0.9*n):]

train_eps = [episodes[i] for i in train_idx]
val_eps   = [episodes[i] for i in val_idx]
test_eps  = [episodes[i] for i in test_idx]

# flatten helpers
def bc_flatten_split_images(episode_dicts, use_skill=True):
    X, y = [], []
    s_key = 'skill_states' if use_skill else 'other_states'
    a_key = 'skill_actions' if use_skill else 'other_actions'
    for ep in episode_dicts:
        X.append(ep[s_key])  # [Ni, H, W, 3]
        y.append(ep[a_key])  # [Ni]
    if len(X) == 0:
        return np.empty((0,274,274,3), dtype=np.float32), np.empty((0,), dtype=int)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

X_tr, y_tr = bc_flatten_split_images(train_eps, use_skill=True)
X_va, y_va = bc_flatten_split_images(val_eps,   use_skill=True)
X_te, y_te = bc_flatten_split_images(test_eps,  use_skill=True)

print(X_tr.shape, y_tr.shape)
print(X_va.shape, y_va.shape)
print(X_te.shape, y_te.shape)

mean_tr, std_tr = compute_channel_mean_std(X_tr)
print("train mean:", mean_tr, "train std:", std_tr)

# constants / device
n_actions = 16
use_mps = torch.backends.mps.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if use_mps else 'cpu'))

# -------------- Build dataloaders from knobs --------------
def build_loaders(batch_size, augment=False):
    normalizer = ImageNormalizer(mean_tr, std_tr)
    train_ds = ImageBCDataset(X_tr, y_tr, normalizer=normalizer, augment=augment)
    val_ds   = ImageBCDataset(X_va, y_va, normalizer=normalizer, augment=False)
    test_ds  = ImageBCDataset(X_te, y_te, normalizer=normalizer, augment=False)

    counts = np.bincount(y_tr, minlength=n_actions).astype(np.float64)
    inv = np.zeros_like(counts); obs = counts > 0
    inv[obs] = 1.0 / counts[obs]
    sample_w = inv[y_tr]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    pin = not use_mps
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, pin_memory=pin, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=max(128, batch_size), shuffle=False, pin_memory=pin, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=max(128, batch_size), shuffle=False, pin_memory=pin, num_workers=4)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

# -------------- Single-run training (also used inside Optuna objective) --------------
def train_eval_once(
    lr=3e-4,
    weight_decay=1e-4,
    batch_size=64,
    label_smoothing=0.0,
    optimizer_name="adamw",
    cosine=False,
    width_mult=1.0,
    drop=0.0,
    max_epochs=60,
    patience=10,
    save_path=default_ckpt_path,
    trial=None,
):
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(batch_size=batch_size, augment=False)

    model = PolicyCNN(n_actions=n_actions, width_mult=width_mult, drop=drop).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if cosine:
        # cosine to 10% of initial lr
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr*0.1)
    else:
        sched = None

    best_val = float('inf'); best_state = None; bad=0
    for epoch in range(max_epochs):
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
            val_acc = correct / max(1, len(val_ds))

        if sched is not None:
            sched.step()

        print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")

        # report to optuna + prune if needed
        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                'state_dict': best_state,
                'mean': mean_tr,
                'std': std_tr,
                'n_actions': n_actions,
                'skill': skill,
                'arch': 'PolicyCNN',
                'epoch': epoch,
                'val_loss': best_val,
                'hparams': {
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'label_smoothing': label_smoothing,
                    'optimizer': optimizer_name,
                    'cosine': cosine,
                    'width_mult': width_mult,
                    'drop': drop,
                }
            }, save_path)
        else:
            bad += 1
            if bad >= patience:
                break

    # evaluate best on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        tot, correct = 0.0, 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        test_nll = tot / max(1, len(test_ds))
        test_acc = correct / max(1, len(test_ds))
    print(f"TEST  NLL {test_nll:.4f} | ACC {test_acc:.3f}")

    return best_val, test_nll, test_acc, save_path

# -------------- Optuna objective --------------
def objective(trial: optuna.Trial):
    # Search space
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    cosine = trial.suggest_categorical("cosine", [False, True])
    width_mult = trial.suggest_float("width_mult", 0.75, 1.5)
    drop = trial.suggest_float("dropout", 0.0, 0.2)
    patience = trial.suggest_int("patience", max(5, args.patience//2), max(20, args.patience*2))

    save_path = os.path.join(ckpt_dir, f'{skill}_policy_cnn_trial{trial.number}.pt')

    best_val, test_nll, test_acc, _ = train_eval_once(
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        optimizer_name=optimizer_name,
        cosine=cosine,
        width_mult=width_mult,
        drop=drop,
        max_epochs=args.epochs,
        patience=patience,
        save_path=save_path,
        trial=trial,
    )

    # Log extra metrics for dashboarding
    trial.set_user_attr("test_nll", float(test_nll))
    trial.set_user_attr("test_acc", float(test_acc))
    return best_val  # minimize val loss

# -------------- Entry points --------------
if not args.study:
    # original single-run behavior
    print("Running a single training job (no study).")
    best_val, test_nll, test_acc, ckpt_path = train_eval_once(
        lr=3e-4,
        weight_decay=1e-4,
        batch_size=64,
        label_smoothing=0.0,
        optimizer_name="adamw",
        cosine=False,
        width_mult=1.0,
        drop=0.0,
        max_epochs=args.epochs,
        patience=args.patience,
        save_path=default_ckpt_path,
        trial=None,
    )
    print(f"Loaded best model. Checkpoint saved at: {ckpt_path}")
else:
    # Optuna study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(3, args.patience//3)) if args.prune else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name=f"{skill}_policy_cnn_study")
    print(f"Starting Optuna study for skill '{skill}' with {args.trials} trials...")
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    print("\n=== Optuna Results ===")
    print(f"Best value (val_loss): {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    # Where the winning checkpoint was saved (by its trial number)
    print(f"Winner checkpoint: {os.path.join(ckpt_dir, f'{skill}_policy_cnn_trial{study.best_trial.number}.pt')}")