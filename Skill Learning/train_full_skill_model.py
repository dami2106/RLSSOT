
import os
import numpy as np
from single_class_svm import create_svm_model, create_svm_model_robust, predict_across_skills


dir_ = '../Data/stone_pick_random_pixels_big'
files = os.listdir(dir_ + '/groundTruth')


def get_unique_skills(files):
    unique_skills = set()
    for file in files:
        with open(os.path.join(dir_ + '/groundTruth', file), 'r') as f:
            lines = f.read().splitlines()
        unique_skills.update(lines)
    return unique_skills
def segment_edges(lst, mode="start"):
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
    skill_states = []
    other_states = [] 

    start_states = []
    end_states = []

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

        # collect start & end feature vectors
        for s in skill_indices:
            skill_states.append(pca_feats[s].tolist())

        starts = segment_edges(skill_indices, mode="start")
        ends   = segment_edges(skill_indices, mode="end")

        for s in starts:
            start_states.append(pca_feats[s].tolist())
        for e in ends:
            end_states.append(pca_feats[e].tolist())

        # everything else (exclude only starts and ends)
        excluded = set(skill_indices)
        for idx in range(n_frames):
            if idx not in excluded:
                other_states.append(pca_feats[idx].tolist())

    return (
        np.array(skill_states),
        np.array(other_states),
        np.array(start_states),
        np.array(end_states)
    )


skills = get_unique_skills(files)
skill_data = {}
for skill in skills:
    skill_states, other_states, start_states, end_states = get_start_end_states(skill, files)
    skill_data[skill] = {
        'skill_states': skill_states,
        'other_states': other_states,
        'start_states': start_states,
        'end_states': end_states
    }


svm_models = {}

for skill, data in skill_data.items():
    skill_states = data['skill_states']
    other_states = data['other_states']
    combined_other = np.concatenate((skill_states, other_states), axis=0)

    print("Start model for skill", skill)
    svm_models[skill] = create_svm_model_robust(skill_states, combined_other)
    print("=" * 60)


for skill in skill_data: 
    #Test the start and end models, add up correct vs incorrect vs total 
    start_states = skill_data[skill]['start_states']
    end_states = skill_data[skill]['end_states']

    correct_start = 0 
    correct_end = 0
    total_start = 0
    total_end = 0

    for feat in start_states:
        predictions = predict_across_skills(feat, svm_models)
        if predictions['top_skill'] == skill:
            correct_start += 1
        total_start += 1

    for feat in end_states:
        predictions = predict_across_skills(feat, svm_models)
        if predictions['top_skill'] == skill:
            correct_end += 1
        total_end += 1

    print(f"Skill: {skill}")
    print(f"Start Model - Correct: {correct_start}, Total: {total_start}")
    print(f"End Model - Correct: {correct_end}, Total: {total_end}")
    print("=" * 60)


