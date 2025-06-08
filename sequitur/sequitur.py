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


def visualize_tree(tree_dict, filename_base, fmt="png", root_label=None):
    """
    tree_dict: nested dict with keys
      - 'production' → int
      - OR 'symbol' → str
      and optional 'children': [ … ]
    filename_base: e.g. "seq_tree_0"  (no extension)
    fmt: "png" or "pdf"
    root_label: if given, use this string as the label for the very top node
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
            label = f"R{node['production']}" if "production" in node else node["symbol"]

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
    # if the last sequence didn’t end with a Mark:
    if current:
        seq_expansions.append(current)

    assert len(seq_expansions) == len(sequences), \
        f"expected {len(sequences)} splits, got {len(seq_expansions)}"

    # assemble one tree per original sequence
    trees = [build_sequence_tree(grammar, exp) for exp in seq_expansions]

    return grammar, trees



def main():
    parser = argparse.ArgumentParser(description='Build HTN hierarchy from sequences')
    parser.add_argument('--data-dir', type=Path, required=True, help='Path to input data directory')
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (data_dir / 'hierarchy_data').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dict = get_unique_sequence_list(data_dir)
    sequences = list(sequence_dict)

    grammar, trees = build_htn_trees(sequences)

    grammar_path = output_dir / 'grammar.txt'
    with grammar_path.open('w') as f:
        for head, productions in grammar.items():
            f.write(f"{head}: {productions}\n")

    for idx, (seq, tree) in enumerate(zip(sequences, trees)):
        png_path = output_dir / f'seq_tree_{idx}'
        visualize_tree(tree, png_path, fmt='png', root_label=f'H{idx}')
        json_path = output_dir / f'tree_{idx}.json'
        json_path.write_text(json.dumps(tree, indent=2))

if __name__ == '__main__':
    main()