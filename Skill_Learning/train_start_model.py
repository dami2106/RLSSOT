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
import matplotlib.pyplot as plt
import seaborn as sns
from skill_helpers import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle

dir_ = 'Data/stone_pick_random_pixels_big'
files = os.listdir(dir_ + '/groundTruth')

SEED = 0

rng = np.random.default_rng(SEED)
results = {}
skills = get_unique_skills(dir_, files)
skill_data = {}

for skill in skills:
    start_states, end_states, all_skill_states, negative_end_skill, \
        negative_end_all, all_other_states = get_start_end_states(dir_, skill, files)
    
    # skill_data[skill] = {
    #     'start_states': start_states,
    #     'end_states': end_states,
    #     'all_skill_states': all_skill_states,
    #     'negative_end_skill': negative_end_skill,
    #     'negative_end_all': negative_end_all,
    #     'all_other_states': all_other_states
    # }

    # [s1, s2, s3],   [s4, s5],    [s6, s7, s8, s9]
    # skill_1         skill_2      skill_1

    # for skill_1, 
    # start_states     = [s1, s6]
    # end_states       = [s3, s9]
    # all_skill_states = [s1, s2, s3,   s6, s7, s8, s9]

    # negative_end_skill = [s1, s2, s6, s7, s8]
    # negative_end_all = [s1, s2, s4, s5, s6, s7, s8]
    # all_other_states = [s4, s5]

    positive_states = negative_end_skill
    negative_states = np.concatenate((end_states, all_other_states))
    X = np.vstack([positive_states, negative_states])
    y = np.hstack([np.ones(len(positive_states), dtype=int), np.zeros(len(negative_states), dtype=int)])
    X, y = shuffle(X, y, random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    print(f"Skill: {skill}")

    print("train balance:", np.bincount(y_train))
    print("test  balance:", np.bincount(y_test))


    base_pipe = make_pipeline(
    StandardScaler(),
    LinearSVC(class_weight="balanced", dual = 'auto', max_iter=50000, random_state=SEED)
    )
    clf = CalibratedClassifierCV(
        estimator=base_pipe,
        method="sigmoid",  
        cv=5                
    )

    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    print("min/max prob:", proba_test.min(), proba_test.max())
    for thr in [0.5, 0.4, 0.3, 0.2, 0.1]:
        preds = (proba_test >= thr).astype(int)
        print(thr, preds.sum())

    y_pred = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    results[skill] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    print(f"Results: {results[skill]}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("="*50+"\n")

