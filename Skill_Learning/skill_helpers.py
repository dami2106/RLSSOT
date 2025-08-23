import os
import numpy as np
import argparse
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
import joblib

def get_unique_skills(files):
    unique_skills = set()
    for file in files:
        with open(os.path.join(dir_ + '/groundTruth', file), 'r') as f:
            lines = f.read().splitlines()
        unique_skills.update(lines)
    return unique_skills

def segment_edges(lst, mode):
    if not lst:
        return []

    if mode not in {"start", "end"}:
        raise ValueError("mode must be 'start' or 'end'")

    edges = []
    seg_start = lst[0]

    for i in range(1, len(lst) + 1):
        if i == len(lst) or lst[i] != lst[i - 1] + 1:
            # segment ended at lst[i-1]
            if mode == "start":
                edges.append(seg_start)
            else:  # mode == "end"
                edges.append(lst[i - 1])
            # prepare for next segment
            if i < len(lst):
                seg_start = lst[i]

    return edges

def get_start_end_states(skill, files):
    start_states = []
    end_states = []
    other_states = []  # new

    for file in files:
        with open(os.path.join(dir_ + '/groundTruth', file), 'r') as f:
            lines = f.read().splitlines()

        pca_feats = np.load(os.path.join(dir_ + '/pca_features', file + '.npy'))
        n_frames = len(pca_feats)
        # assert n_frames == len(lines), f"Mismatch in {file}: {n_frames} feats vs {len(lines)} labels"

        # indices where this skill appears
        skill_indices = [i for i, x in enumerate(lines) if x == skill]
        if not skill_indices:
            # if the skill never occurs, then *all* frames are "other"
            other_states.extend(pca_feats.tolist())
            continue

        # starts and ends for this skill's contiguous segments
        starts = segment_edges(skill_indices, mode="start")
        ends   = segment_edges(skill_indices, mode="end")

        # collect start & end feature vectors
        for s in starts:
            start_states.append(pca_feats[s].tolist())
        for e in ends:
            end_states.append(pca_feats[e].tolist())

        # everything else (exclude only starts and ends)
        excluded = set(starts) | set(ends)
        for idx in range(n_frames):
            if idx not in excluded:
                other_states.append(pca_feats[idx].tolist())

    return (
        np.array(start_states),
        np.array(end_states),
        np.array(other_states),
    )

def evaluate_ocsvm(X_pos_train, X_pos_val, X_neg_val, nu, gamma):
    # Scale + fit on positive-only training split
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                        OneClassSVM(kernel="rbf", nu=nu, gamma=gamma))
    clf.fit(X_pos_train)

    # Predict on validation (pos + neg)
    X_val = np.vstack([X_pos_val, X_neg_val])
    y_true = np.hstack([np.ones(len(X_pos_val), dtype=int),
                        np.zeros(len(X_neg_val), dtype=int)])
    y_pred = clf.predict(X_val)
    y_pred = (y_pred == 1).astype(int)  # map {+1,-1} -> {1,0}

    # Use F1 of the positive class as selection metric
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    return f1, clf


def choose_skill_given_state(state, start_models):
    best_skill = None
    best_score = -np.inf

    for skill, model in start_models.items():
        score = model.decision_function(state.reshape(1, -1))
        if score > best_score:
            best_score = score
            best_skill = skill

    return best_skill, best_score


# def check_if_end_skill(state, end_model_skill):
  
#     score = end_model_skill.decision_function(state.reshape(1, -1))

#     return score

def check_if_end_skill(state, end_model_skill, threshold=0.0):
    """
    Returns (is_end, score) where:
      - is_end = True if state is an inlier to the end-model
      - score  = decision_function value (higher = more inlier)
    """
    score = float(end_model_skill.decision_function(state.reshape(1, -1)))
    return (score > threshold), score