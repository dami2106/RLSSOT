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

def get_unique_skills(dir_, files):
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



# [s1, s2, s3],   [s4, s5],    [s6, s7, s8, s9]
# skill_1         skill_2      skill_1

# for skill_1, 
# start_states     = [s1, s6]
# end_states       = [s3, s9]
# all_skill_states = [s1, s2, s3,   s6, s7, s8, s9]

# negative_end_skill = [s1, s2, s6, s7, s8]
# negative_end_all = [s1, s2, s4, s5, s6, s7, s8]

def get_start_end_states(dir_, skill, files):
    start_states = []
    end_states = []
    all_skill_states = []

    negative_end_skill = []
    negative_end_all = []
    all_other_states = []

    for file in files:
        with open(os.path.join(dir_, 'groundTruth', file), 'r') as f:
            lines = f.read().splitlines()

        pca_feats = np.load(os.path.join(dir_, 'pca_features', file + '.npy'))

        # keep them in sync in case of off-by-one labeling issues
        n = min(len(lines), len(pca_feats))
        lines = lines[:n]
        pca_feats = pca_feats[:n]

        # indices where this skill appears
        skill_indices = [i for i, x in enumerate(lines) if x == skill]

        # starts and ends for this skill's contiguous segments
        starts = segment_edges(skill_indices, mode="start")
        ends   = segment_edges(skill_indices, mode="end")

        # for quick membership tests
        ends_set = set(ends)

        # collect start & end feature vectors for this skill
        for s in starts:
            start_states.append(pca_feats[s].tolist())
        for e in ends:
            end_states.append(pca_feats[e].tolist())

        # all frames of this skill
        for i in skill_indices:
            all_skill_states.append(pca_feats[i].tolist())

        # negative_end_skill: all skill frames except the skill's end frames
        for i in skill_indices:
            if i not in ends_set:
                negative_end_skill.append(pca_feats[i].tolist())

        # negative_end_all: all frames (any label) except the skill's end frames
        for i in range(n):
            if i not in ends_set:
                negative_end_all.append(pca_feats[i].tolist())

        # all_other_states: all frames NOT belonging to this skill
        for i in range(n):
            if lines[i] != skill:
                all_other_states.append(pca_feats[i].tolist())

    return (
        np.array(start_states),
        np.array(end_states),
        np.array(all_skill_states),
        np.array(negative_end_skill),
        np.array(negative_end_all),
        np.array(all_other_states),
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