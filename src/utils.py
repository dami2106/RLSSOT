import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
import wandb

from metrics import pred_to_gt_match, filter_exclusions
import os


def plot_segmentation_gt(gt, pred, mask, gt_uniq=None, pred_to_gt=None, name=''):
    colors = {}

    pred_, gt_ = filter_exclusions(pred[mask].cpu().numpy(), gt[mask].cpu().numpy())
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_, gt_)
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())
    for pr_lab, gt_lab in zip(pred_opt, gt_opt):
        pred_[pred_ == pr_lab] = gt_lab
    n_frames = len(pred_)

    # add colors for predictions which do not match to a gt class

    if gt_uniq is None:
        gt_uniq = np.unique(gt_.cpu().numpy())
    pred_not_matched = np.setdiff1d(pred_opt, gt_uniq)
    if len(pred_not_matched) > 0:
        gt_uniq = np.concatenate((gt_uniq, pred_not_matched))

    n_class = len(gt_uniq)
    if n_class <= 20:
        cmap = plt.get_cmap('tab20')
    else:  # up to 40 classes
        cmap1 = plt.get_cmap('tab20')
        cmap2 = plt.get_cmap('tab20b')
        cmap = lambda x: cmap1(round(x * n_class / 20., 2)) if x <= 19. / n_class else cmap2(round((x - 20 / n_class) * n_class / 20, 2))

    for i, label in enumerate(gt_uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / n_class)

    fig = plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(name, fontsize=45, pad=20)

    # plot gt segmentation

    ax = fig.add_subplot(2, 1, 1)
    ax.set_ylabel('TRUTH', fontsize=30, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    gt_segment_boundaries = np.where(gt_[1:] - gt_[:-1])[0] + 1
    gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries, [len(gt_)]))

    for start, end in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
        label = gt_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    # plot predicted segmentation after matching to gt labels w/Hungarian

    ax = fig.add_subplot(2, 1, 2)
    ax.set_ylabel('ASOT', fontsize=30, rotation=0, labelpad=60, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred_[1:] - pred_[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred_)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig


def plot_segmentation(pred, mask, name=''):
    colors = {}
    cmap = plt.get_cmap('tab20')
    uniq = np.unique(pred[mask].cpu().numpy())
    n_frames = len(pred)

    # add colors for predictions which do not match to a gt class

    for i, label in enumerate(uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / len(uniq))

    fig = plt.figure(figsize=(16, 2))
    plt.axis('off')
    plt.title(name, fontsize=30, pad=20)

    # plot gt segmentation

    ax = fig.add_subplot(1, 1, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred[mask].cpu().numpy()[1:] - pred[mask].cpu().numpy()[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred[mask].cpu().numpy()[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig


def plot_matrix(mat, gt=None, colorbar=True, title=None, figsize=(10, 5), ylabel=None, xlabel=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot1 = ax.matshow(mat)
    if gt is not None: # plot gt segment boundaries
        gt_change = np.where((np.diff(gt) != 0))[0] + 1
        for ch in gt_change:
            ax.axvline(ch, color='red')
    if colorbar:
        plt.colorbar(plot1, ax=ax)
    if title:
        ax.set_title(f'{title}')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=36)
        ax.tick_params(axis='x', labelsize=24)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=36)
        ax.tick_params(axis='y', labelsize=24)
    ax.set_aspect('auto')
    fig.tight_layout()
    return fig


def save_skill_ordering(skills, fname, out_dir="skill_orderings"):
    """
    Save the skill ordering (skills at each time step) to a text file.
    Args:
        skills: 1D numpy array or list of skill indices (length = num_frames)
        fname:  Name of the episode/file (used for naming the output file)
        out_dir: Directory to save the text files
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname}_skills.txt")
    with open(out_path, "w") as f:
        for skill in skills:
            f.write(f"{skill}\n")


def save_matching_mapping(pred_to_gt, out_dir="skill_orderings"):
    """
    Save the Hungarian matching (predicted to GT mapping) to a text file.
    Args:
        pred_to_gt: dict or list of (pred_label, gt_label) pairs
        out_dir: Directory to save the mapping file
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mapping.txt")
    with open(out_path, "w") as f:
        if isinstance(pred_to_gt, dict):
            items = pred_to_gt.items()
        else:
            items = pred_to_gt
        for pred_label, gt_label in items:
            f.write(f"{pred_label} {gt_label}\n")