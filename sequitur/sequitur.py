from sksequitur import Parser, Grammar, Production, Mark
from graphviz import Digraph
import argparse
from helpers import get_unique_sequence_list
import argparse
import json
from pathlib import Path
import os
from structure_metrics import compute_structure_metrics
from similarity_metrics import compute_similarity_metrics
from matching.games import StableMarriage

# --- 3) Build a nested‐dict tree for one sequence‐expansion ---
def build_sequence_tree(grammar, token_list):
    """
    Turn a list of tokens [Prod, 'A', Prod, …] into
    {'production': 0, 'children': [ … ]} recursively.
    """
    node = {"production": 0, "children": []}
    for tok in token_list:
        if isinstance(tok, Production):
            node["children"].append(build_subtree(grammar, tok))
        else:
            node["children"].append({"symbol": tok})
    return node

def build_subtree(grammar, prod):
    """
    Turn a Production(prod) into {'production': n, 'children': […]}.
    Uses the single right‐hand side grammar[prod].
    """
    subtree = {"production": int(prod), "children": []}
    for tok in grammar[prod]:
        if isinstance(tok, Production):
            subtree["children"].append(build_subtree(grammar, tok))
        else:
            subtree["children"].append({"symbol": tok})
    return subtree


def visualize_tree(tree_dict, filename_base, fmt="png", root_label=None, label_mapping=None):
    """
    tree_dict: nested dict with keys
      - 'production' → int
      - OR 'symbol' → str
      and optional 'children': [ … ]
    filename_base: e.g. "seq_tree_0"  (no extension)
    fmt: "png" or "pdf"
    root_label: if given, use this string as the label for the very top node
    label_mapping: optional dict mapping grammar labels to readable strings
    """
    dot = Digraph(format=fmt)
    dot.attr(rankdir="TB")     # top→bottom
    dot.attr("node", shape="oval", fontsize="12")

    def recurse(node, parent_id=None, is_root=False):
        nid = str(id(node))
        # if this is the root of the entire tree and a custom label was supplied, use it:
        if is_root and root_label is not None:
            label = root_label
        else:
            if "production" in node:
                label = f"R{node['production']}"
            else:
                # Use mapping if available, otherwise use original symbol
                label = label_mapping.get(node["symbol"], node["symbol"]) if label_mapping else node["symbol"]

        dot.node(nid, label)
        if parent_id is not None:
            dot.edge(parent_id, nid)

        for child in node.get("children", []):
            recurse(child, nid)

    # kick off recursion marking the first call as the root
    recurse(tree_dict, parent_id=None, is_root=True)

    outpath = dot.render(filename_base, cleanup=True)
    print(f"Written {outpath}")



def build_htn_trees(sequences):
    """
    Build HTN trees for each sequence in the list.
    sequences: list of lists of symbols (not strings)
    """
    # Create a single parser instance
    parser = Parser()

    # Feed each sequence to the parser incrementally
    for seq in sequences:
        # seq is now a list of symbols, not a string
        parser.feed(seq)
        parser.feed([Mark()])

    # Convert the parser output to a grammar
    grammar = Grammar(parser.tree)

    # Print the final grammar
    print(grammar)

    flat_start = grammar[Production(0)]
    seq_expansions = []
    current = []
    for tok in flat_start:
        if isinstance(tok, Mark):
            seq_expansions.append(current)
            current = []
        else:
            current.append(tok)
    # if the last sequence didn't end with a Mark:
    if current:
        seq_expansions.append(current)

    assert len(seq_expansions) == len(sequences), \
        f"expected {len(sequences)} splits, got {len(seq_expansions)}"

    # assemble one tree per original sequence
    trees = [build_sequence_tree(grammar, exp) for exp in seq_expansions]

    return grammar, trees



def construct_hierarchy(args, dataset_dir, hierarchy_output_dir, skill_folder, mapping, vis_mapping):
  
    sequence_dict = get_unique_sequence_list(dataset_dir, skill_folder, mapping)
    sequences = [list(seq) for seq, count in sequence_dict.items() if count >= args.threshold]
    sequences = [list(seq) for seq in sequence_dict]

    grammar, trees = build_htn_trees(sequences)

    grammar_path = hierarchy_output_dir / 'grammar.txt'
    with grammar_path.open('w') as f:
        for head, productions in grammar.items():
            f.write(f"{head}: {productions}\n")

    # Dictionary to store structure metrics for all trees
    structure_metrics_dict = {}

    for idx, (seq, tree) in enumerate(zip(sequences, trees)):
        png_path = hierarchy_output_dir / f'seq_tree_{idx}'
        visualize_tree(tree, png_path, fmt='png', root_label=f'H{idx}', label_mapping=vis_mapping)
        json_path = hierarchy_output_dir / f'tree_{idx}.json'
        json_path.write_text(json.dumps(tree, indent=2))
        
        # Compute structure metrics for this tree
        metrics = compute_structure_metrics(tree)
        structure_metrics_dict[f'tree_{idx}'] = metrics

    # Save structure metrics to JSON file
    metrics_path = hierarchy_output_dir / 'structure_metrics.json'
    with metrics_path.open('w') as f:
        json.dump(structure_metrics_dict, f, indent=2)
    
    print(f"Structure metrics saved to {metrics_path}")
    
    return trees

def compute_all_similarity_metrics(gt_trees, pred_trees, output_dir):
    """
    Compute similarity metrics between all pairs of ground truth and predicted trees.
    Returns a dictionary with keys (gt_tree_idx, pred_tree_idx) and similarity metrics as values.
    """
    similarity_metrics_dict = {}
    
    for gt_idx, gt_tree in enumerate(gt_trees):
        for pred_idx, pred_tree in enumerate(pred_trees):
            pair_key = f"(seq_tree_{gt_idx}_gt, seq_tree_{pred_idx}_pred)"
            metrics = compute_similarity_metrics(gt_tree, pred_tree)
            similarity_metrics_dict[pair_key] = metrics
    
    # Save similarity metrics to JSON file
    similarity_path = output_dir / 'similarity_metrics.json'
    with similarity_path.open('w') as f:
        json.dump(similarity_metrics_dict, f, indent=2)
    
    print(f"Similarity metrics saved to {similarity_path}")
    return similarity_metrics_dict

def perform_stable_marriage_matching(similarity_metrics_dict, gt_count, pred_count, output_dir):
    """
    Perform matching between ground truth and predicted trees.
    Uses similarity metrics to create optimal matches.
    Handles cases where there are different numbers of trees on each side.
    """
    print(f"Ground truth trees: {gt_count}, Predicted trees: {pred_count}")
    
    # Calculate the maximum tree edit distance for normalization
    max_tree_edit_distance = 0
    for pair_key, metrics in similarity_metrics_dict.items():
        max_tree_edit_distance = max(max_tree_edit_distance, metrics["tree_edit_distance"])
    
    print(f"Maximum tree edit distance found: {max_tree_edit_distance}")
    
    # If we have equal numbers, we can use stable marriage
    if gt_count == pred_count:
        # Create preference lists for both sides
        gt_preferences = {}
        pred_preferences = {}
        
        # Initialize preference dictionaries
        for gt_idx in range(gt_count):
            gt_preferences[f"gt_{gt_idx}"] = []
        for pred_idx in range(pred_count):
            pred_preferences[f"pred_{pred_idx}"] = []
        
        # Build preference lists based on similarity metrics
        for gt_idx in range(gt_count):
            for pred_idx in range(pred_count):
                pair_key = f"(seq_tree_{gt_idx}_gt, seq_tree_{pred_idx}_pred)"
                if pair_key in similarity_metrics_dict:
                    metrics = similarity_metrics_dict[pair_key]
                    # Create a composite score (higher is better)
                    # Use dynamic max_tree_edit_distance for normalization
                    normalized_ted = 1.0 - (metrics["tree_edit_distance"] / max_tree_edit_distance) if max_tree_edit_distance > 0 else 0.0
                    composite_score = (
                        metrics["jaccard_index"] * 0.4 +
                        metrics["subtree_overlap"] * 0.4 +
                        normalized_ted * 0.2
                    )
                    
                    gt_preferences[f"gt_{gt_idx}"].append((f"pred_{pred_idx}", composite_score))
                    pred_preferences[f"pred_{pred_idx}"].append((f"gt_{gt_idx}", composite_score))
        
        # Sort preferences by score (descending)
        for gt_key in gt_preferences:
            gt_preferences[gt_key].sort(key=lambda x: x[1], reverse=True)
            gt_preferences[gt_key] = [pred_key for pred_key, _ in gt_preferences[gt_key]]
        
        for pred_key in pred_preferences:
            pred_preferences[pred_key].sort(key=lambda x: x[1], reverse=True)
            pred_preferences[pred_key] = [gt_key for gt_key, _ in pred_preferences[pred_key]]
        
        # Perform stable marriage matching
        matching = StableMarriage(gt_preferences, pred_preferences)
        matches = matching.solve()
        
    else:
        # Use greedy matching for unequal numbers
        print("Using greedy matching due to unequal tree counts")
        matches = perform_greedy_matching(similarity_metrics_dict, gt_count, pred_count, max_tree_edit_distance)
    
    # Convert matches to a more readable format
    matching_results = {}
    for gt_key, pred_key in matches.items():
        gt_idx = int(gt_key.split('_')[1])
        pred_idx = int(pred_key.split('_')[1])
        pair_key = f"(seq_tree_{gt_idx}_gt, seq_tree_{pred_idx}_pred)"
        
        if pair_key in similarity_metrics_dict:
            matching_results[pair_key] = {
                "similarity_metrics": similarity_metrics_dict[pair_key],
                "gt_index": gt_idx,
                "pred_index": pred_idx
            }
    
    # Save matching results to JSON file
    matching_path = output_dir / 'stable_marriage_matching.json'
    with matching_path.open('w') as f:
        json.dump(matching_results, f, indent=2)
    
    print(f"Matching results saved to {matching_path}")
    print(f"Found {len(matches)} optimal matches")
    
    return matching_results

def perform_greedy_matching(similarity_metrics_dict, gt_count, pred_count, max_tree_edit_distance):
    """
    Perform greedy matching between ground truth and predicted trees.
    This handles cases where there are different numbers of trees.
    """
    # Create a list of all possible pairs with their scores
    pairs = []
    for gt_idx in range(gt_count):
        for pred_idx in range(pred_count):
            pair_key = f"(seq_tree_{gt_idx}_gt, seq_tree_{pred_idx}_pred)"
            if pair_key in similarity_metrics_dict:
                metrics = similarity_metrics_dict[pair_key]
                # Create a composite score (higher is better)
                # Use dynamic max_tree_edit_distance for normalization
                normalized_ted = 1.0 - (metrics["tree_edit_distance"] / max_tree_edit_distance) if max_tree_edit_distance > 0 else 0.0
                composite_score = (
                    metrics["jaccard_index"] * 0.4 +
                    metrics["subtree_overlap"] * 0.4 +
                    normalized_ted * 0.2
                )
                pairs.append((f"gt_{gt_idx}", f"pred_{pred_idx}", composite_score))
    
    # Sort pairs by score (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching: take the best available pair at each step
    matches = {}
    used_gt = set()
    used_pred = set()
    
    for gt_key, pred_key, score in pairs:
        if gt_key not in used_gt and pred_key not in used_pred:
            matches[gt_key] = pred_key
            used_gt.add(gt_key)
            used_pred.add(pred_key)
    
    return matches


def main():
    parser = argparse.ArgumentParser(description='Build HTN hierarchy from sequences')
    parser.add_argument('--predicted-dir', type=Path, required=True, help='Path to input data directory')
    parser.add_argument('--dataset-dir', type=Path, default='Traces/stone_pick_random/stone_pick_random_pixels', help='Name of the folder containing the skill sequences')
    parser.add_argument('--threshold', type=int, default=2, help='Minimum frequency of a sequence to be considered (number of times it appears in the data)')
    args = parser.parse_args()
    
    predicted_dir = args.predicted_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (predicted_dir / 'hierarchy_data').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    pred_hierarchy_output_dir = output_dir / 'predicted_hierarchy'
    pred_hierarchy_output_dir.mkdir(parents=True, exist_ok=True)

    gt_hierarchy_output_dir = output_dir / 'ground_truth_hierarchy'
    gt_hierarchy_output_dir.mkdir(parents=True, exist_ok=True)

    matched_hierarchy_output_dir = output_dir / 'matched_hierarchy'
    matched_hierarchy_output_dir.mkdir(parents=True, exist_ok=True)
    


    #Stores the hungarian matching map for predicted -> intermediate
    mapping_path = predicted_dir / 'mapping' / 'mapping.txt'
    with mapping_path.open('r') as f:
        mapping = {}
        for line in f:
            pred_label, gt_label = line.strip().split()
            mapping[pred_label] = gt_label
    
    #Stores the ground truth mapping for intermediate -> truth
    gt_mapping_path = dataset_dir / 'mapping' / 'mapping.txt'
    with gt_mapping_path.open('r') as f:
        gt_mapping = {}
        for line in f:
            pred_label, gt_label = line.strip().split()
            gt_mapping[pred_label] = gt_label

    #Combine both mappings to get predicted -> truth
    pred_mapping = {}
    for pred_label, intermediate_label in mapping.items():
        if intermediate_label in gt_mapping:
            pred_mapping[pred_label] = gt_mapping[intermediate_label]
        else:
            print(f"Warning: intermediate label '{intermediate_label}' not found in gt_mapping")

    pred_trees = construct_hierarchy(args, predicted_dir, pred_hierarchy_output_dir, 'predicted_skills', None, None)

    gt_trees = construct_hierarchy(args, dataset_dir, gt_hierarchy_output_dir, 'groundTruth', gt_mapping, gt_mapping)

    matched_trees = construct_hierarchy(args, predicted_dir, matched_hierarchy_output_dir, 'predicted_skills', None, pred_mapping)

    similarity_metrics_dict = compute_all_similarity_metrics(gt_trees, pred_trees, output_dir)

    perform_stable_marriage_matching(similarity_metrics_dict, len(gt_trees), len(pred_trees), output_dir)


if __name__ == '__main__':
    main()