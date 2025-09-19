import os
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from skill_helpers import *  # uses build_endability_dataset etc.
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier


import optuna
_HAS_OPTUNA = True


SEED = 42
dir_ = 'Craftax/Traces/stone_pickaxe_easy'

# Directory to save trained end models & metadata
models_dir = os.path.join(dir_, 'pu_end_models_tuned')
os.makedirs(models_dir, exist_ok=True)
files = os.listdir(os.path.join(dir_, 'groundTruth'))

# ----------------------------
# Helpers
# ----------------------------
def best_threshold_from_pr(y_true, p_scores):
    """Map max-F1 point back to thresholds correctly (thresholds align with prec[1:], rec[1:])."""
    prec, rec, thr = precision_recall_curve(y_true, p_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)

    if len(thr) == 0:  # degenerate case
        best_idx = int(np.nanargmax(f1s))
        return 0.5, float(f1s[best_idx])

    valid = f1s[1:]
    best_idx = int(np.nanargmax(valid)) + 1
    return float(thr[best_idx - 1]), float(f1s[best_idx])


def make_pu_clf(
    method: str = "elkanoto",   # "elkanoto" or "bagging"
    C: float = 10.0,
    kernel: str = "rbf",        # "linear" is faster; "rbf" often stronger
    gamma: str | float = "scale",
    hold_out_ratio: float = 0.2,  # used by Elkanoto
    n_estimators: int = 15,       # used by BaggingPuClassifier
    seed: int = SEED,
):
    """
    Build a PU-learning estimator that exposes predict_proba.
    y must be 1 for positive, 0 for unlabeled (your current y fits this).
    """
    base = make_pipeline(
        StandardScaler(),
        SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=seed)
    )

    if method.lower() == "bagging":
        pu = BaggingPuClassifier(base_estimator=base, n_estimators=n_estimators, random_state=seed)
    else:
        pu = ElkanotoPuClassifier(estimator=base, hold_out_ratio=hold_out_ratio, random_state=seed)
    return pu


def evaluate_config_on_groupval(X, y, groups, *, cfg, seed=SEED):
    """
    Threshold selection by group-held-out validation; returns model, val F1, and chosen threshold.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(gss.split(X, y, groups))
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    pu = make_pu_clf(
        method=cfg["method"], C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"],
        hold_out_ratio=cfg["hold_out_ratio"], n_estimators=cfg["n_estimators"], seed=seed
    )
    pu.fit(X_tr, y_tr)
    val_proba = pu.predict_proba(X_val)[:, 1]
    thr, val_f1 = best_threshold_from_pr(y_val, val_proba)

    return {"model": pu, "threshold": float(thr), "val_f1": float(val_f1)}


def refit_on_all_train(X, y, *, cfg, seed=SEED):
    pu_full = make_pu_clf(
        method=cfg["method"], C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"],
        hold_out_ratio=cfg["hold_out_ratio"], n_estimators=cfg["n_estimators"], seed=seed
    )
    pu_full.fit(X, y)
    return pu_full


# ----------------------------
# Hyperparameter search
# ----------------------------
def _search_space():
    return {
        "method": ["elkanoto", "bagging"],
        "C": {"low": 1e-2, "high": 1e3},
        "kernel": ["rbf", "linear"],
        "gamma_kind": ["scale", "auto", "numeric"],   # numeric only if kernel=rbf
        "gamma_numeric": {"low": 1e-4, "high": 10.0},
        "hold_out_ratio": {"low": 0.1, "high": 0.4},  # Elkanoto-only
        "n_estimators": {"low": 5, "high": 50},       # Bagging-only
    }

def _random_sample(space, rng):
    method = rng.choice(space["method"])
    kernel = rng.choice(space["kernel"])
    C = float(np.exp(np.log(space["C"]["low"]) + rng.random() *
                     (np.log(space["C"]["high"]) - np.log(space["C"]["low"]))))

    if kernel == "rbf":
        gk = rng.choice(space["gamma_kind"])
        if gk == "numeric":
            g_low, g_high = space["gamma_numeric"]["low"], space["gamma_numeric"]["high"]
            gamma = float(np.exp(np.log(g_low) + rng.random() * (np.log(g_high) - np.log(g_low))))
        else:
            gamma = gk  # "scale" or "auto"
    else:
        gamma = "scale"

    hold_out_ratio = float(
        space["hold_out_ratio"]["low"] + rng.random() *
        (space["hold_out_ratio"]["high"] - space["hold_out_ratio"]["low"])
    )
    n_estimators = int(rng.integers(space["n_estimators"]["low"], space["n_estimators"]["high"] + 1))

    return {
        "method": method,
        "C": C,
        "kernel": kernel,
        "gamma": gamma,
        "hold_out_ratio": hold_out_ratio,
        "n_estimators": n_estimators,
    }

def tune_pu_svc(X, y, groups, *, n_trials=40, timeout_secs=None, seed=SEED, use_optuna=True):
    """
    Returns (best_cfg, best_thr, best_val_f1).
    Uses Optuna if available; otherwise random sweep over the same space.
    """
    space = _search_space()

    if use_optuna and _HAS_OPTUNA:
        def objective(trial):
            method = trial.suggest_categorical("method", space["method"])
            kernel = trial.suggest_categorical("kernel", space["kernel"])
            C = trial.suggest_float("C", space["C"]["low"], space["C"]["high"], log=True)

            if kernel == "rbf":
                gk = trial.suggest_categorical("gamma_kind", space["gamma_kind"])
                if gk == "numeric":
                    gamma = trial.suggest_float("gamma_numeric", space["gamma_numeric"]["low"], space["gamma_numeric"]["high"], log=True)
                else:
                    gamma = gk
            else:
                gamma = "scale"

            if method == "elkanoto":
                hold_out_ratio = trial.suggest_float("hold_out_ratio", space["hold_out_ratio"]["low"], space["hold_out_ratio"]["high"])
                n_estimators = 15
            else:
                hold_out_ratio = 0.2
                n_estimators = trial.suggest_int("n_estimators", space["n_estimators"]["low"], space["n_estimators"]["high"])

            cfg = {
                "method": method, "C": float(C), "kernel": kernel, "gamma": gamma,
                "hold_out_ratio": float(hold_out_ratio), "n_estimators": int(n_estimators),
            }

            try:
                out = evaluate_config_on_groupval(X, y, groups, cfg=cfg, seed=seed)
                # Maximize F1 -> minimize negative
                return -out["val_f1"]
            except Exception:
                return 1e9

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed), study_name="pu_end_svc")
        study.optimize(objective, n_trials=n_trials, timeout=timeout_secs)

        p = study.best_params
        method = p.get("method", "elkanoto")
        kernel = p.get("kernel", "rbf")
        C = float(p.get("C", 10.0))
        if kernel == "rbf":
            gk = p.get("gamma_kind", "scale")
            gamma = float(p.get("gamma_numeric", 0.1)) if gk == "numeric" else gk
        else:
            gamma = "scale"
        if method == "elkanoto":
            hold_out_ratio = float(p.get("hold_out_ratio", 0.2))
            n_estimators = 15
        else:
            hold_out_ratio = 0.2
            n_estimators = int(p.get("n_estimators", 15))

        best_cfg = {
            "method": method, "C": C, "kernel": kernel, "gamma": gamma,
            "hold_out_ratio": hold_out_ratio, "n_estimators": n_estimators,
            "optuna_best_val_f1": float(-study.best_value),
        }
        out = evaluate_config_on_groupval(X, y, groups, cfg=best_cfg, seed=seed)
        return best_cfg, float(out["threshold"]), float(out["val_f1"])

    # -------- Randomized fallback --------
    rng_local = np.random.default_rng(seed)
    best = {"f1": -1.0, "thr": 0.5, "cfg": None}
    for _ in range(n_trials):
        cfg = _random_sample(space, rng_local)
        try:
            out = evaluate_config_on_groupval(X, y, groups, cfg=cfg, seed=seed)
            if out["val_f1"] > best["f1"]:
                best = {"f1": out["val_f1"], "thr": out["threshold"], "cfg": cfg}
        except Exception:
            continue
    if best["cfg"] is None:
        return {"method": "elkanoto", "C": 10.0, "kernel": "rbf", "gamma": "scale", "hold_out_ratio": 0.2, "n_estimators": 15}, 0.5, 0.0
    return best["cfg"], float(best["thr"]), float(best["f1"])


# ----------------------------
# Train & evaluate per skill (with tuning)
# ----------------------------
results = {}
skills = get_unique_skills(dir_, files)

# Tuning controls (adjust as needed)
N_TRIALS = 50
TIMEOUT_SECS = None     # e.g., 900 for a 15-minute cap

for skill in skills:
    X, y, groups = build_endability_dataset(dir_, skill, files, features_dirname='pca_features_750')

    # Reproducible permutation
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    X, y, groups = X[perm], y[perm], groups[perm]

    # Group-aware outer split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]

    print(f"Skill: {skill}")
    print("train balance:", np.bincount(y_train))
    print("test  balance:",  np.bincount(y_test))

    # ---- Hyperparameter tuning (PU + SVC) ----
    best_cfg, thr, val_f1 = tune_pu_svc(
        X_train, y_train, groups_train,
        n_trials=N_TRIALS, timeout_secs=TIMEOUT_SECS, seed=SEED, use_optuna=True
    )
    print("[TUNING] Best cfg:", best_cfg)
    print("[TUNING] Chosen threshold (val):", thr, " (val F1=", f"{val_f1:.4f}", ")")

    # ---- Refit best on all training data ----
    clf = refit_on_all_train(X_train, y_train, cfg=best_cfg, seed=SEED)

    # ---- Evaluate on test using tuned threshold ----
    proba_test = clf.predict_proba(X_test)[:, 1]
    print("min/max prob:", float(proba_test.min()), float(proba_test.max()))
    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
        print(t, int((proba_test >= t).sum()))
    y_pred = (proba_test >= thr).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    results[skill] = {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "val_f1": float(val_f1),
        "best_cfg": best_cfg,
    }

    # Persist model + metadata
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path  = os.path.join(models_dir, f"{skill}_meta.json")
    try:
        dump(clf, model_path)
        with open(meta_path, 'w') as f:
            meta = {
                'skill': skill,
                'threshold': float(thr),
                'val_f1': float(val_f1),
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1': float(f1),
                'n_train_pos': int((y_train == 1).sum()),
                'n_train_neg': int((y_train == 0).sum()),
                'n_test_pos': int((y_test == 1).sum()),
                'n_test_neg': int((y_test == 0).sum()),
                'seed': SEED,
                'tuned': True,
                'best_params': best_cfg,   # includes method, C, kernel, gamma, etc.
            }
            if "optuna_best_val_f1" in best_cfg:
                meta["optuna_best_val_f1"] = float(best_cfg["optuna_best_val_f1"])
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save model/metadata for skill {skill}: {e}")

# ----------------------------
# Aggregated reporting
# ----------------------------
rows = []
tot_tn = tot_fp = tot_fn = tot_tp = 0

for skill, res in results.items():
    cm = res["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    tot_tn += int(tn); tot_fp += int(fp); tot_fn += int(fn); tot_tp += int(tp)

    support_pos = int(tp + fn)
    support_neg = int(tn + fp)
    support_all = support_pos + support_neg
    acc = (tp + tn) / support_all if support_all else float("nan")

    rows.append({
        "skill": skill,
        "pos_support": support_pos,
        "neg_support": support_neg,
        "threshold": res["threshold"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "accuracy": acc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })

overall_support = tot_tp + tot_fp + tot_fn + tot_tn
overall_precision = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) else 0.0
overall_recall    = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) else 0.0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) else 0.0
overall_accuracy  = (tot_tp + tot_tn) / overall_support if overall_support else float("nan")

macro_precision = float(np.mean([r["precision"] for r in rows])) if rows else float("nan")
macro_recall    = float(np.mean([r["recall"]    for r in rows])) if rows else float("nan")
macro_f1        = float(np.mean([r["f1"]        for r in rows])) if rows else float("nan")
macro_accuracy  = float(np.mean([r["accuracy"]  for r in rows])) if rows else float("nan")

print("\n" + "="*80)
print("PER-SKILL METRICS (sorted by F1 desc)")
print("="*80)
df = pd.DataFrame(rows).sort_values("f1", ascending=False)
disp_cols = ["skill", "pos_support", "neg_support", "threshold",
             "precision", "recall", "f1", "accuracy", "tp", "fp", "fn"]
for c in ["threshold", "precision", "recall", "f1", "accuracy"]:
    df[c] = df[c].astype(float).round(3)
print(df[disp_cols].to_string(index=False))

metrics_csv  = os.path.join(models_dir, 'per_skill_metrics.csv')
metrics_json = os.path.join(models_dir, 'summary_metrics.json')
try:
    df.to_csv(metrics_csv, index=False)
    with open(metrics_json, 'w') as f:
        json.dump({
            'overall': {
                'support': int(overall_support),
                'tp': int(tot_tp), 'fp': int(tot_fp), 'fn': int(tot_fn), 'tn': int(tot_tn),
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1': float(overall_f1),
                'accuracy': float(overall_accuracy)
            },
            'macro': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1': float(macro_f1),
                'accuracy': float(macro_accuracy)
            }
        }, f, indent=2)
except Exception as e:
    print(f"[WARN] Failed to save aggregate metrics: {e}")

print("\n" + "="*80)
print("OVERALL (MICRO) METRICS — pooled over all skills")
print("="*80)
print(f"Support (all skills): {overall_support}")
print(f"TP={tot_tp}  FP={tot_fp}  FN={tot_fn}  TN={tot_tn}")
print(f"Precision: {overall_precision:.3f}  Recall: {overall_recall:.3f}  F1: {overall_f1:.3f}  Accuracy: {overall_accuracy:.3f}")

print("\n" + "="*80)
print("MACRO AVERAGES — mean of per-skill metrics")
print("="*80)
print(f"Precision: {macro_precision:.3f}  Recall: {macro_recall:.3f}  F1: {macro_f1:.3f}  Accuracy: {macro_accuracy:.3f}")
print("="*80 + "\n")