This repository contains a modified version of the **Action Segmentation Optimal Transport (ASOT)** model, originally presented in the [CVPR 2024 paper](http://arxiv.org/abs/2404.01518):  
**"Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation"**.

> 📌 **Note**: This is a *modification* of the original ASOT implementation, adapted to work with **reinforcement learning (RL) trajectories** instead of video data. The core methodology remains the same.

## 1. Overview: Action Segmentation Optimal Transport (ASOT)

ASOT is an unsupervised segmentation technique that decodes a set of temporally consistent pseudo-labels from a noisy affinity matrix. In this implementation, ASOT is adapted to operate over RL trajectories rather than video frame features.

It uses a **structure-aware optimal transport** formulation to enforce:

- **Temporal consistency** via Gromov-Wasserstein optimal transport
- **Robustness to class imbalance** via unbalanced OT

ASOT is used within an unsupervised learning framework where representations and action cluster embeddings are jointly learned using pseudo-labels generated per batch.

## 2. Using the Code

### Run Training

We provide a single training script that handles the full training and evaluation pipeline:

```bash
bash training.sh
```
All necessary hyperparameters and flags are set within this script.

## 3. Dataset Format
data/                 
├─ dataset_name/                # e.g., rl_dataset/
│  ├─ features/                 # per-timestep features (Each file should be 2-dimensional)
│  │  ├─ traj1.npy              
│  │  ├─ traj2.npy              
│  │  ├─ ...                    
│  ├─ groundTruth/              # per-timestep labels for evaluation
│  │  ├─ traj1                  
│  │  ├─ traj2                  
│  │  ├─ ...                    
│  ├─ mapping/                 
│  │  ├─ mapping.txt            # maps skills to IDs 

You can define your own dataset_name (e.g., rl_dataset) and use the above structure to add new datasets.


