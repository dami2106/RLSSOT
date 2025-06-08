from sksequitur import Parser, Grammar, Production, Mark
from graphviz import Digraph
import argparse
from helpers import get_unique_sequence_list
import argparse
import json
from pathlib import Path
import os

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
    """
    # Create a single parser instance
    parser = Parser()

    # Feed each sequence to the parser incrementally
    for seq in sequences:
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
    sequences = [seq for seq, count in sequence_dict.items() if count >= args.threshold]
    sequences = list(sequence_dict)

    grammar, trees = build_htn_trees(sequences)

    grammar_path = hierarchy_output_dir / 'grammar.txt'
    with grammar_path.open('w') as f:
        for head, productions in grammar.items():
            f.write(f"{head}: {productions}\n")

    for idx, (seq, tree) in enumerate(zip(sequences, trees)):
        png_path = hierarchy_output_dir / f'seq_tree_{idx}'
        visualize_tree(tree, png_path, fmt='png', root_label=f'H{idx}', label_mapping=vis_mapping)
        json_path = hierarchy_output_dir / f'tree_{idx}.json'
        json_path.write_text(json.dumps(tree, indent=2))


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

    construct_hierarchy(args, predicted_dir, pred_hierarchy_output_dir, 'predicted_skills', None, None)

    construct_hierarchy(args, dataset_dir, gt_hierarchy_output_dir, 'groundTruth', gt_mapping, gt_mapping)

    construct_hierarchy(args, predicted_dir, matched_hierarchy_output_dir, 'predicted_skills', None, pred_mapping)





if __name__ == '__main__':
    main()