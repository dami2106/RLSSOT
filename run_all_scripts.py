#!/usr/bin/env python3

import os
import subprocess
import re
import csv


def parse_metrics(output):
    """
    Parse the metrics of interest from the combined stdout/stderr of a script run.
    Returns a dict with keys: f1_full, f1_per, miou_full, miou_per, mof_full, mof_per
    """
    patterns = {
        'f1_full': r'test_f1_full\s+([0-9.]+)',
        'f1_per': r'test_f1_per\s+([0-9.]+)',
        'miou_full': r'test_miou_full\s+([0-9.]+)',
        'miou_per': r'test_miou_per\s+([0-9.]+)',
        'mof_full': r'test_mof_full\s+([0-9.]+)',
        'mof_per': r'test_mof_per\s+([0-9.]+)',
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


def main():
    base_dir = 'Scripts'
    csv_file = 'scripts_output.csv'
    fieldnames = ['script_name', 'f1_full', 'f1_per', 'miou_full', 'miou_per', 'mof_full', 'mof_per']

    # Open CSV and write header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over subfolders in base_dir
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # Iterate over shell scripts in each subfolder
            for filename in os.listdir(subdir_path):
                if not filename.endswith('.sh'):
                    continue
                script_path = os.path.join(subdir_path, filename)
                print(f'Running {script_path}...')
                os.chmod(script_path, 0o755)

                print(f"Script path is {script_path}")

                # Run the shell script without raising on non-zero exit
                result = subprocess.run(
                    ['bash', script_path],
                    capture_output=True,
                    text=True
                )

                # Warn if script exited with error
                if result.returncode != 0:
                    print(f'Warning: {script_path} exited with status {result.returncode}')

                # Combine stdout and stderr for parsing
                output = (result.stdout or '') + (result.stderr or '')

                # Parse the metrics from the output
                metrics = parse_metrics(output)
                row = {
                    'script_name': filename,
                    'f1_full': metrics.get('f1_full'),
                    'f1_per': metrics.get('f1_per'),
                    'miou_full': metrics.get('miou_full'),
                    'miou_per': metrics.get('miou_per'),
                    'mof_full': metrics.get('mof_full'),
                    'mof_per': metrics.get('mof_per'),
                }

                # Write the parsed metrics to the CSV
                writer.writerow(row)
                print(f'Parsed metrics for {filename}: {row}')


if __name__ == '__main__':
    main()
