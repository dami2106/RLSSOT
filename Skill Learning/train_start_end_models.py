
import os
import numpy as np
from oc_svm import OneClassSVMClassifier


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


skills = get_unique_skills(files)
skill_data = {}
for skill in skills:
    start_states, end_states, other_states = get_start_end_states(skill, files)
    skill_data[skill] = {
        'start_states': start_states,
        'end_states': end_states,
        'other_states': other_states
    }


svm_start_models = {}
for skill, data in skill_data.items():
    start_states = data['start_states']
    end_states = data['end_states']
    other_states = data['other_states']

    # Train start state model using OC SVM
    if len(start_states) > 0:
        print(f"Training start state model for skill: {skill}")
        print(f"  - Start states: {len(start_states)}")
        print(f"  - End states: {len(end_states)}")
        print(f"  - Other states: {len(other_states)}")
        
        # Create and train start state model
        start_model = OneClassSVMClassifier(kernel='rbf', nu=0.1, gamma='scale', verbose=False)
        
        # For start states, we want to learn what constitutes a valid start
        # We'll use start_states as positive examples (inliers)
        start_model.fit(start_states)
        
        # Store the trained model
        svm_start_models[skill] = start_model
        
        print(f"  ✓ Start state model trained successfully")
    else:
        print(f"Warning: No start states found for skill: {skill}")
        svm_start_models[skill] = None

svm_end_models = {}
for skill, data in skill_data.items():
    start_states = data['start_states']
    end_states = data['end_states']
    other_states = data['other_states']

    # Train end state model using OC SVM
    if len(end_states) > 0:
        print(f"Training end state model for skill: {skill}")
        print(f"  - Start states: {len(start_states)}")
        print(f"  - End states: {len(end_states)}")
        print(f"  - Other states: {len(other_states)}")
        
        # Create and train end state model
        end_model = OneClassSVMClassifier(kernel='rbf', nu=0.1, gamma='scale', verbose=False)
        
        # For end states, we want to learn what constitutes a valid end
        # We'll use end_states as positive examples (inliers)
        end_model.fit(end_states)
        
        # Store the trained model
        svm_end_models[skill] = end_model
        
        print(f"  ✓ End state model trained successfully")
    else:
        print(f"Warning: No end states found for skill: {skill}")
        svm_end_models[skill] = None

print(f"\nTraining completed!")
print(f"Start state models trained: {len([m for m in svm_start_models.values() if m is not None])}")
print(f"End state models trained: {len([m for m in svm_end_models.values() if m is not None])}")

# Function to predict if a state is a start state for a given skill
def is_start_state(state_features, skill, models):
    """Check if the given state is a start state for the specified skill."""
    if skill not in models or models[skill] is None:
        return False
    
    # Reshape if needed (single sample)
    if state_features.ndim == 1:
        state_features = state_features.reshape(1, -1)
    
    # Predict using the OC SVM (1 = inlier/start state, -1 = outlier/not start state)
    prediction = models[skill].predict(state_features)
    return prediction[0] == 1

# Function to predict if a state is an end state for a given skill
def is_end_state(state_features, skill, models):
    """Check if the given state is an end state for the specified skill."""
    if skill not in models or models[skill] is None:
        return False
    
    # Reshape if needed (single sample)
    if state_features.ndim == 1:
        state_features = state_features.reshape(1, -1)
    
    # Predict using the OC SVM (1 = inlier/end state, -1 = outlier/not end state)
    prediction = models[skill].predict(state_features)
    return prediction[0] == 1

# Function to get confidence scores for start/end state predictions
def get_start_state_confidence(state_features, skill, models):
    """Get confidence score for start state prediction (higher = more confident it's a start state)."""
    if skill not in models or models[skill] is None:
        return 0.0
    
    # Reshape if needed (single sample)
    if state_features.ndim == 1:
        state_features = state_features.reshape(1, -1)
    
    # Get decision function score (higher positive values = more confident it's a start state)
    confidence = models[skill].decision_function(state_features)
    return confidence[0]

def get_end_state_confidence(state_features, skill, models):
    """Get confidence score for end state prediction (higher = more confident it's an end state)."""
    if skill not in models or models[skill] is None:
        return 0.0
    
    # Reshape if needed (single sample)
    if state_features.ndim == 1:
        state_features = state_features.reshape(1, -1)
    
    # Get decision function score (higher positive values = more confident it's an end state)
    confidence = models[skill].decision_function(state_features)
    return confidence[0]

# Example usage of the trained models
print(f"\nExample predictions:")
for skill in list(skills)[:3]:  # Show first 3 skills as examples
    if skill in svm_start_models and svm_start_models[skill] is not None:
        # Test with a sample start state
        sample_start = skill_data[skill]['start_states'][0]
        is_start = is_start_state(sample_start, skill, svm_start_models)
        start_conf = get_start_state_confidence(sample_start, skill, svm_start_models)
        
        print(f"Skill '{skill}' - Sample start state:")
        print(f"  Is start state: {is_start}")
        print(f"  Start confidence: {start_conf:.4f}")
        
        # Test with a sample end state
        if len(skill_data[skill]['end_states']) > 0:
            sample_end = skill_data[skill]['end_states'][0]
            is_end = is_end_state(sample_end, skill, svm_end_models)
            end_conf = get_end_state_confidence(sample_end, skill, svm_end_models)
            
            print(f"  Is end state: {is_end}")
            print(f"  End confidence: {end_conf:.4f}")
        print()

# Save all trained models
def save_all_models(save_dir='../Data/trained_models'):
    """Save all trained start and end state models to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save start state models
    start_models_dir = os.path.join(save_dir, 'start_models')
    os.makedirs(start_models_dir, exist_ok=True)
    
    for skill, model in svm_start_models.items():
        if model is not None:
            model_path = os.path.join(start_models_dir, f'{skill}_start.joblib')
            model.save_model(model_path)
            print(f"Saved start model for '{skill}' to {model_path}")
    
    # Save end state models
    end_models_dir = os.path.join(save_dir, 'end_models')
    os.makedirs(end_models_dir, exist_ok=True)
    
    for skill, model in svm_end_models.items():
        if model is not None:
            model_path = os.path.join(end_models_dir, f'{skill}_end.joblib')
            model.save_model(model_path)
            print(f"Saved end model for '{skill}' to {model_path}")
    
    print(f"\nAll models saved to {save_dir}")

# Load all trained models
def load_all_models(load_dir='../Data/trained_models'):
    """Load all trained start and end state models from disk."""
    loaded_start_models = {}
    loaded_end_models = {}
    
    # Load start state models
    start_models_dir = os.path.join(load_dir, 'start_models')
    if os.path.exists(start_models_dir):
        for model_file in os.listdir(start_models_dir):
            if model_file.endswith('_start.joblib'):
                skill = model_file.replace('_start.joblib', '')
                model_path = os.path.join(start_models_dir, model_file)
                loaded_start_models[skill] = OneClassSVMClassifier.load_model(model_path)
                print(f"Loaded start model for '{skill}' from {model_path}")
    
    # Load end state models
    end_models_dir = os.path.join(load_dir, 'end_models')
    if os.path.exists(end_models_dir):
        for model_file in os.listdir(end_models_dir):
            if model_file.endswith('_end.joblib'):
                skill = model_file.replace('_end.joblib', '')
                model_path = os.path.join(end_models_dir, model_file)
                loaded_end_models[skill] = OneClassSVMClassifier.load_model(model_path)
                print(f"Loaded end model for '{skill}' from {model_path}")
    
    return loaded_start_models, loaded_end_models

# Function to find the best skill to execute based on current state
def find_best_start_skill(current_state, models, confidence_threshold=0.0):
    """
    Find the best skill to start based on the current state.
    
    Args:
        current_state: Current state features
        models: Dictionary of trained start state models
        confidence_threshold: Minimum confidence threshold for considering a skill
        
    Returns:
        Tuple of (best_skill, confidence) or (None, 0.0) if no suitable skill found
    """
    best_skill = None
    best_confidence = confidence_threshold
    
    for skill, model in models.items():
        if model is not None:
            confidence = get_start_state_confidence(current_state, skill, models)
            if confidence > best_confidence:
                best_confidence = confidence
                best_skill = skill
    
    return best_skill, best_confidence

def find_best_end_skill(current_state, models, confidence_threshold=0.0):
    """
    Find the best skill that has reached its end state based on the current state.
    
    Args:
        current_state: Current state features
        models: Dictionary of trained end state models
        confidence_threshold: Minimum confidence threshold for considering a skill
        
    Returns:
        Tuple of (best_skill, confidence) or (None, 0.0) if no suitable skill found
    """
    best_skill = None
    best_confidence = confidence_threshold
    
    for skill, model in models.items():
        if model is not None:
            confidence = get_end_state_confidence(current_state, skill, models)
            if confidence > best_confidence:
                best_confidence = confidence
                best_skill = skill
    
    return best_skill, best_confidence

# Function to evaluate model performance on test data
def evaluate_models_on_data(test_data, start_models, end_models):
    """
    Evaluate the performance of start and end state models on test data.
    
    Args:
        test_data: Dictionary with skill data containing start_states, end_states, other_states
        start_models: Dictionary of trained start state models
        end_models: Dictionary of trained end state models
        
    Returns:
        Dictionary containing evaluation metrics for each skill
    """
    results = {}
    
    for skill in test_data.keys():
        if skill not in start_models or skill not in end_models:
            continue
            
        start_model = start_models[skill]
        end_model = end_models[skill]
        
        if start_model is None or end_model is None:
            continue
        
        # Test start state detection
        start_states = test_data[skill]['start_states']
        end_states = test_data[skill]['end_states']
        other_states = test_data[skill]['other_states']
        
        # Start state evaluation
        start_correct = 0
        start_total = len(start_states)
        
        for state in start_states:
            if is_start_state(state, skill, start_models):
                start_correct += 1
        
        # End state evaluation
        end_correct = 0
        end_total = len(end_states)
        
        for state in end_states:
            if is_end_state(state, skill, end_models):
                end_correct += 1
        
        # False positive evaluation (other states incorrectly classified as start/end)
        start_fp = 0
        end_fp = 0
        
        for state in other_states:
            if is_start_state(state, skill, start_models):
                start_fp += 1
            if is_end_state(state, skill, end_models):
                end_fp += 1
        
        results[skill] = {
            'start_accuracy': start_correct / start_total if start_total > 0 else 0,
            'end_accuracy': end_correct / end_total if end_total > 0 else 0,
            'start_false_positives': start_fp,
            'end_false_positives': end_fp,
            'start_total': start_total,
            'end_total': end_total
        }
    
    return results

# Save models by default
print("\nSaving trained models...")
save_all_models()

print("\n" + "="*60)
print("Training and setup complete!")
print("="*60)
print("\nYou can now use these models for:")
print("1. Detecting when to start a skill (option initiation)")
print("2. Detecting when a skill has completed (option termination)")
print("3. Behavioral cloning pipeline integration")
print("\nExample usage:")
print("  # Check if current state is a start state for 'mine_stone'")
print("  is_start = is_start_state(current_state, 'mine_stone', svm_start_models)")
print("  ")
print("  # Get confidence scores")
print("  start_conf = get_start_state_confidence(current_state, 'mine_stone', svm_start_models)")
print("  end_conf = get_end_state_confidence(current_state, 'mine_stone', svm_end_models)")
print("  ")
print("  # Find best skill to execute")
print("  best_skill, confidence = find_best_start_skill(current_state, svm_start_models)")


