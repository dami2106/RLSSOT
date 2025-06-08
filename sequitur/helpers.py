import os 
from collections import Counter


def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                key, value = line.strip().split(maxsplit=1)
                mapping[int(key)] = value

    reverse_mapping = {v: chr(ord('A') + k) for k, v in mapping.items()}
    return reverse_mapping



def get_unique_sequence_list(trace_directory, skill_folder = "predicted_skills"):
    """
    Get a list of unique sequences from the trace directory.
    """
    sequence_list = []


    for filename in os.listdir(f"{trace_directory}/{skill_folder}/"):
        with open(os.path.join(f"{trace_directory}/{skill_folder}/", filename), 'r') as file:
            sequence = file.read().strip().split()

            result = []
            prev = None
            for item in sequence:
                if item != prev:
                    result.append(item)
                    prev = item

            sequence_list.append("".join([skill for skill in result]))
    
    counts = Counter(lst for lst in sequence_list)

    return counts
    


if __name__ == "__main__":
    trace_directory = "runs/stone_pick_random/stone_pick_random_pixels_stone_pick_random_pixels/version_0"
    counts = get_unique_sequence_list(trace_directory)
    print(counts)

    