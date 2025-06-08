import os 
from collections import Counter

#Give the directory of ground truth eg Cobblestone/groundTruth
def get_sequence_list(trace_directory):
    """
    Get a list of sequences from the trace directory.
    """
    sequence_list = []
    for filename in os.listdir(trace_directory):
        with open(os.path.join(trace_directory, filename), 'r') as file:
            sequence = file.read().strip().split()
            sequence_list.append(sequence)
    return sequence_list

def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                key, value = line.strip().split(maxsplit=1)
                mapping[int(key)] = value

    reverse_mapping = {v: chr(ord('A') + k) for k, v in mapping.items()}
    return reverse_mapping



def get_unique_sequence_list(trace_directory):
    """
    Get a list of unique sequences from the trace directory.
    """
    sequence_list = []

    mapping = load_mapping(f"{trace_directory}/mapping/mapping.txt")

    for filename in os.listdir(f"{trace_directory}/groundTruth/"):
        with open(os.path.join(f"{trace_directory}/groundTruth/", filename), 'r') as file:
            sequence = file.read().strip().split()

            #Get unique elements in the sequence maintaining the order
            seen = set()
            unique_sequence = []
            for item in sequence:
                if item not in seen:
                    seen.add(item)
                    unique_sequence.append(item)
            
            sequence_list.append("".join([mapping[skill] for skill in unique_sequence]))
    
    counts = Counter(lst for lst in sequence_list)

    return counts
    


