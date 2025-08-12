import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score
from scipy.stats import loguniform
import numpy as np
class SingleClassSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=random_state)

    def fit(self, X_pos, X_neg):
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_pos, X_neg):
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        print('Confusion Matrix:')
        print(confusion_matrix(y, y_pred))
        print('\nClassification Report:')
        print(classification_report(y, y_pred, digits=4))
        print(f'ROC AUC: {roc_auc_score(y, y_proba):.4f}')


def create_svm_model(start_states, all_other_states, test_size = 0.1):
    X_pos_train, X_pos_test = train_test_split(start_states, test_size=test_size, random_state=42)
    X_neg_train, X_neg_test = train_test_split(all_other_states, test_size=test_size, random_state=42)

    svm = SingleClassSVM()
    svm.fit(X_pos_train, X_neg_train)
    print('Evaluation on test set:')
    svm.evaluate(X_pos_test, X_neg_test)

    return svm

def create_svm_model_robust(start_states, all_other_states, test_size=0.1, neg_pos_ratio=3, random_state=42):


    # X, y
    X_pos = np.asarray(start_states)
    X_neg = np.asarray(all_other_states)

    # Downsample negatives to control imbalance (cap at neg_pos_ratio * positives)
    rng = np.random.default_rng(random_state)
    max_neg = min(len(X_neg), neg_pos_ratio * len(X_pos))
    if len(X_neg) > max_neg:
        idx = rng.choice(len(X_neg), size=max_neg, replace=False)
        X_neg = X_neg[idx]

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos), dtype=int), np.zeros(len(X_neg), dtype=int)])

    # Stratified train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, test_idx), = sss.split(X, y)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Pipeline: scale -> SVC (probabilities for threshold tuning)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
    ])

    # Quick hyperparam search (log-uniform for C and gamma)
    param_dist = {
        "svc__C": loguniform(1e-2, 1e3),
        "svc__gamma": loguniform(1e-4, 1e0),
    }
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist,
        n_iter=25, cv=3, scoring="f1", n_jobs=-1, random_state=random_state, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # Tune probability threshold on a validation split inside train
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    (tr_idx, val_idx), = sss_val.split(X_train, y_train)
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    model.fit(X_tr, y_tr)

    # Choose threshold that maximizes F1 on validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    candidate_thresholds = np.quantile(val_proba, np.linspace(0.05, 0.95, 19))
    best_thresh, best_f1 = 0.5, -1.0
    for t in candidate_thresholds:
        f1 = f1_score(y_val, (val_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    # Refit on full train (with best hyperparams)
    model.fit(X_train, y_train)

    # Evaluate on test with tuned threshold
    test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= best_thresh).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report_txt = classification_report(y_test, y_pred, digits=4)
    roc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)

    print("Evaluation on test set:")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_txt)
    print(f"ROC AUC: {roc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print(f"Chosen threshold: {best_thresh:.4f}")

    return model, best_thresh, {
        "confusion_matrix": cm,
        "classification_report": report_txt,
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }



def predict_across_skills(feature, svm_models):
    x = np.asarray(feature).reshape(1, -1)

    probs  = {}
    binary = {}
    scores = {}

    for skill, pack in svm_models.items():
        # unpack (support both tuple and plain model)
        if isinstance(pack, tuple) and len(pack) >= 2:
            model, thresh = pack[0], pack[1]
        else:
            model, thresh = pack, 0.5  # default threshold if not stored

        # probability of positive class ("start of this skill")
        p = float(model.predict_proba(x)[:, 1][0])
        probs[skill] = p

        # raw SVM margin (useful for tie-breaks / interpretability)
        if hasattr(model, "decision_function"):
            scores[skill] = float(model.decision_function(x)[0])
        else:
            scores[skill] = float(p)  # fallback

        # binary decision using per-model tuned threshold
        binary[skill] = int(p >= thresh)

    ranking = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_skill, top_prob = ranking[0]

    return {
        "probs": probs,
        "binary": binary,
        "scores": scores,
        "ranking": ranking,
        "top_skill": top_skill,
        "top_prob": top_prob,
    }

if __name__ == "__main__":
    np.random.seed(42)
    start_states = np.random.randn(500, 650) + 0.5 
    all_other_states = np.random.randn(2000, 650)

    # Split into train/test
    create_svm_model(start_states, all_other_states)
