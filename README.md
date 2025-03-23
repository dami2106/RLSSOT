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

## 3. Dataset Format

The expected dataset format follows the original ASOT structure and consists of:

```
data/                 
├─ dataset_name/                # e.g., rl_dataset/
│  ├─ features/                 # pre-extracted per-frame or per-timestep features
│  │  ├─ traj1.npy              
│  │  ├─ traj2.npy              
│  │  ├─ ...                    
│  ├─ groundTruth/              # optional: per-timestep labels for evaluation
│  │  ├─ traj1                  
│  │  ├─ traj2                  
│  │  ├─ ...                    
│  ├─ mapping/                 
│  │  ├─ mapping.txt            # maps action/class IDs to labels
```

You can define your own `dataset_name` (e.g., `rl_dataset`) and use the above structure to add new datasets.

## 4. Dependencies

Install the following packages before running the training script:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pytorch`
- `pytorch-lightning`
- `wandb`

## 5. Reference

This code is based on the original ASOT model from the CVPR 2024 paper:

- **Paper**: [Temporally Consistent Unbalanced Optimal Transport for Unsupervised Action Segmentation](http://arxiv.org/abs/2404.01518)
- **Original Authors**: Ming Xu, Stephen Gould.
- **Original Code**: [ASOT](https://github.com/mingu6/action_seg_ot)


