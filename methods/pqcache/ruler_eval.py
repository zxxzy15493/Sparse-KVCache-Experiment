#!/usr/bin/env python3
"""
Given a benchmark root directory, compute the average SCORE from summary.csv by length.
If a length has no summary.csv, show num.
"""

import csv
import argparse
from pathlib import Path


def parse_summary(summary_path):
    """Parse summary.csv, return the list of values in the SCORE row."""
    with open(summary_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0 and row[0].strip() == 'Score':
                return [float(x) for x in row[1:] if x.strip()]
    return []


def find_length_dirs(root):
    """Find length subdirs, which may be pred directly or contain pred."""
    # root/4096/pred structure
    length_dirs = []
    for d in root.iterdir():
        if d.is_dir():
            sub_pred = d / 'pred'
            if sub_pred.is_dir() and (sub_pred / 'summary.csv').exists():
                length_dirs.append(d.name)
            else:
                # May be a subdir like synthetic, keep searching deeper
                for dd in d.iterdir():
                    if dd.is_dir():
                        sub_sub_pred = dd / 'pred'
                        if sub_sub_pred.is_dir() and (sub_sub_pred / 'summary.csv').exists():
                            length_dirs.append(f"{d.name}/{dd.name}")

    if not length_dirs:
        return [], root
    return sorted(length_dirs, key=lambda x: int(x.split('/')[-1])), root


def get_summary_path(root, length):
    """Get the summary.csv path for the given length."""
    if '/' in length:
        # synthetic/4096 format, needs pred/
        summary = root / length / 'pred' / 'summary.csv'
    else:
        summary = root / length / 'pred' / 'summary.csv'
    if summary.exists():
        return summary
    return None


def main():
    parser = argparse.ArgumentParser(description='Compute benchmark SCORE averages')
    parser.add_argument('root_dir', help='Benchmark root directory path')
    args = parser.parse_args()

    root = Path(args.root_dir)
    lengths, base_root = find_length_dirs(root)

    print(f"{'Length':<20} {'SCORE avg':<15} {'Tasks':<10}")
    print("-" * 48)

    for length in lengths:
        summary_path = get_summary_path(base_root, length)

        if summary_path is None or not summary_path.exists():
            print(f"{length:<20} {'num':<15}")
            continue

        scores = parse_summary(summary_path)
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{length:<20} {avg:<15.2f} {len(scores):<10}")
        else:
            print(f"{length:<20} {'num':<15}")


if __name__ == '__main__':
    main()
