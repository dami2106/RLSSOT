import os
import json
import argparse
import numpy as np
from skill_helpers import *  # noqa: F401,F403
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import optuna
from datetime import datetime
from typing import Sequence, Dict, Any


class Standardizer:
    """Simple numpy feature standardizer (mean/std)"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)

class BCDataset(Dataset):
    """Step-level tuples for MLP returning (state_t, action_t)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class PolicyMLP(nn.Module):
    def __init__(self, d_in: int, n_actions: int = 17, hidden_sizes: Sequence[int] = (256, 128, 64), p_drop: float = 0.1):
        super().__init__()
        layers = [nn.LayerNorm(d_in)]
        dim = d_in
        for h in hidden_sizes:
            layers += [nn.Linear(dim, h), nn.GELU(), nn.Dropout(p_drop)]
            dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_dataloaders(X_tr, y_tr, X_va, y_va, X_te, y_te, batch_size: int, n_actions: int, use_mps: bool):
    scaler = Standardizer()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    train_ds = BCDataset(X_tr, y_tr)
    val_ds = BCDataset(X_va, y_va)
    test_ds = BCDataset(X_te, y_te)

    counts = np.bincount(y_tr, minlength=n_actions).astype(np.float64)
    inv = np.zeros_like(counts)
    obs = counts > 0
    inv[obs] = 1.0 / counts[obs]
    sample_w = inv[y_tr]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, pin_memory=not use_mps)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=not use_mps)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=not use_mps)
    return scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def train_one_model(hparams: Dict[str, Any], X_tr, y_tr, X_va, y_va, X_te, y_te, n_actions: int, device: torch.device, max_patience: int = 10):
    use_mps = torch.backends.mps.is_available()
    scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te, hparams['batch_size'], n_actions, use_mps
    )
    model = PolicyMLP(
        d_in=X_tr.shape[1],
        n_actions=n_actions,
        hidden_sizes=hparams['hidden_sizes'],
        p_drop=hparams['dropout']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

    best_val = float('inf')
    bad = 0
    best_state = None
    for epoch in range(hparams['epochs']):
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

        # validation
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
            val_acc = correct / len(val_ds) if len(val_ds) > 0 else 0.0

        if hparams.get('verbose'):
            print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= max_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test metric
    model.eval()
    with torch.no_grad():
        tot, correct = 0.0, 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        test_loss = tot / len(test_ds)
        test_acc = correct / len(test_ds) if len(test_ds) > 0 else 0.0

    return {
        'model': model,
        'scaler': scaler,
        'val_loss': best_val,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }


def run_optuna_for_skill(skill: str, episodes, n_actions: int, device: torch.device, args):
    print(f"===== OPTUNA STUDY FOR SKILL {skill} =====")

    # Pre-split episodes once to avoid repeating io.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n = len(idx)
    train_idx = idx[: int(0.8 * n)]
    val_idx = idx[int(0.8 * n): int(0.9 * n)]
    test_idx = idx[int(0.9 * n):]
    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]
    test_eps = [episodes[i] for i in test_idx]

    X_tr, y_tr = bc_flatten_split(train_eps, use_skill=True)
    X_va, y_va = bc_flatten_split(val_eps, use_skill=True)
    X_te, y_te = bc_flatten_split(test_eps, use_skill=True)

    def objective(trial: optuna.Trial):
        # search space
        depth = trial.suggest_int('depth', 1, 4)
        base_size = trial.suggest_categorical('base_size', [64, 128, 256, 384])
        decay_factor = trial.suggest_float('decay_factor', 0.3, 0.8)
        hidden_sizes = [int(base_size * (decay_factor ** i)) for i in range(depth)]
        dropout = trial.suggest_float('dropout', 0.0, 0.4)
        lr = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        epochs = trial.suggest_int('epochs', 60, 300)

        hparams = dict(
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            verbose=False,
        )
        result = train_one_model(hparams, X_tr, y_tr, X_va, y_va, X_te, y_te, n_actions, device)
        return result['val_loss']

    study_dir = os.path.join(args.output_dir, f"skill_{skill}")
    os.makedirs(study_dir, exist_ok=True)
    study_path = os.path.join(study_dir, 'optuna_study.pkl')
    if os.path.exists(study_path) and args.resume:
        print("Resuming existing study...")
        try:
            study = optuna.load_study(study_name=f"skill_{skill}", storage=f"sqlite:///{os.path.join(study_dir, 'study.db')}")
        except Exception:
            study = optuna.create_study(direction='minimize')
    else:
        study = optuna.create_study(direction='minimize')

    def save_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial):  # noqa: D401
        df = study.trials_dataframe()
        df.to_csv(os.path.join(study_dir, 'optuna_results.csv'), index=False)
        try:
            import joblib
            joblib.dump(study, study_path)
        except Exception as e:
            print(f"Warning: failed to serialize study: {e}")

    study.optimize(objective, n_trials=args.trials, callbacks=[save_progress], show_progress_bar=not args.no_progress)

    # final save
    save_progress(study, None)
    best_trial = study.best_trial
    print(f"Best val loss {best_trial.value:.4f} with params: {best_trial.params}")

    # retrain best with verbosity and more epochs (optional) if requested
    best_params = best_trial.params.copy()
    hidden_sizes = []
    base_size = best_params['base_size']
    decay_factor = best_params['decay_factor']
    for i in range(best_params['depth']):
        hidden_sizes.append(int(base_size * (decay_factor ** i)))
    hparams = dict(
        hidden_sizes=hidden_sizes,
        dropout=best_params['dropout'],
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs'] if not args.refit_epochs else args.refit_epochs,
        verbose=True,
    )
    final = train_one_model(hparams, X_tr, y_tr, X_va, y_va, X_te, y_te, n_actions, device)
    model = final['model']
    torch.save(model.state_dict(), os.path.join(study_dir, 'best_model.pt'))
    with open(os.path.join(study_dir, 'best_metrics.json'), 'w') as f:
        json.dump({k: v for k, v in final.items() if k not in ['model', 'scaler']}, f, indent=2)
    with open(os.path.join(study_dir, 'best_config.json'), 'w') as f:
        json.dump(hparams, f, indent=2)


def train_fixed_for_skill(skill: str, episodes, n_actions: int, device: torch.device, args):
    print(f"===== TRAINING SKILL {skill} (fixed params) =====")
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n = len(idx)
    train_idx = idx[: int(0.8 * n)]
    val_idx = idx[int(0.8 * n): int(0.9 * n)]
    test_idx = idx[int(0.9 * n):]

    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]
    test_eps = [episodes[i] for i in test_idx]

    X_tr, y_tr = bc_flatten_split(train_eps, use_skill=True)
    X_va, y_va = bc_flatten_split(val_eps, use_skill=True)
    X_te, y_te = bc_flatten_split(test_eps, use_skill=True)

    print(X_tr.shape, y_tr.shape)
    print(X_va.shape, y_va.shape)
    print(X_te.shape, y_te.shape)

    hparams = dict(
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=True,
    )
    result = train_one_model(hparams, X_tr, y_tr, X_va, y_va, X_te, y_te, n_actions, device)
    model = result['model']
    skill_dir = os.path.join(args.output_dir, f"skill_{skill}")
    os.makedirs(skill_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(skill_dir, 'model.pt'))
    with open(os.path.join(skill_dir, 'metrics.json'), 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['model', 'scaler']}, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Behavioural Cloning with optional Optuna HPO")
    p.add_argument('--data-dir', type=str, default='Data/Test', help='Base dataset directory (contains groundTruth etc.)')
    p.add_argument('--feature-name', type=str, default='pca_features', help='Feature folder name used by helper loader')
    p.add_argument('--skills', type=str, nargs='*', default=None, help='Subset of skills to train (default: all)')
    p.add_argument('--output-dir', type=str, default='BC_Results', help='Where to store study results/models')
    p.add_argument('--seed', type=int, default=0)
    # fixed training params (used when not using optuna)
    p.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 128, 64])
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=800)
    # optuna options
    p.add_argument('--use-optuna', action='store_true')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--refit-epochs', type=int, default=None, help='If set, refit best trial for this many epochs')
    p.add_argument('--resume', action='store_true', help='Attempt to resume existing study')
    p.add_argument('--no-progress', action='store_true', help='Disable optuna progress bar')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dir_ = args.data_dir
    files = os.listdir(os.path.join(dir_, 'groundTruth'))
    unique_skills = get_unique_skills(dir_, files)
    if args.skills:
        unique_skills = [s for s in unique_skills if s in args.skills]
    print(f"Found {len(unique_skills)} skills: {unique_skills}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_actions = 16
    for skill in unique_skills:
        episodes = get_bc_data_by_episode(dir_, files, skill, feature_name=args.feature_name)
        if args.use_optuna:
            run_optuna_for_skill(skill, episodes, n_actions, device, args)
        else:
            train_fixed_for_skill(skill, episodes, n_actions, device, args)
        print("========================\n")


if __name__ == '__main__':
    main()


