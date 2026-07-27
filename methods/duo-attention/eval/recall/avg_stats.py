import csv
import sys
import os

if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = os.path.join(os.path.dirname(__file__), "qasper_attn_ratios.csv")

recall_100_list = []
recall_k_list = []
selected_attn_ratio_list = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        recall_100_list.append(float(row["recall_100"]))
        recall_k_list.append(float(row["recall_k"]))
        selected_attn_ratio_list.append(float(row["selected_attn_ratio"]))

n = len(recall_100_list)
print(f"Total rows: {n}")
print()
print(f"recall_100          avg: {sum(recall_100_list) / n:.6f}")
print(f"recall_k            avg: {sum(recall_k_list) / n:.6f}")
print(f"selected_attn_ratio avg: {sum(selected_attn_ratio_list) / n:.6f}")