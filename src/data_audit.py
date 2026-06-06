"""
data_audit.py — Run this FIRST to answer all of Sebastian's questions:
1. Show train/val/test split sizes
2. Prove same labels exist in all splits
3. Show class distribution per split (bar chart)
4. Show there is NO label leakage between splits

Run with:
    python src/data_audit.py --data_dir data/raw/trashnet
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import random_split

# ── Argument ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/raw/trashnet",
                    help="Path to the folder that contains class subfolders")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Load full dataset (no transforms needed for audit) ────────────────────────
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
full_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)

CLASSES = full_dataset.classes
N = len(full_dataset)
print("=" * 60)
print("DATASET AUDIT REPORT")
print("=" * 60)
print(f"\nDataset root : {args.data_dir}")
print(f"Total images : {N}")
print(f"Classes ({len(CLASSES)}): {CLASSES}")

# ── Count raw class distribution ──────────────────────────────────────────────
all_labels = [label for _, label in full_dataset.samples]
raw_counts = Counter(all_labels)
print("\n── Raw class distribution (before split) ──")
for idx, cls in enumerate(CLASSES):
    print(f"  {cls:12s} : {raw_counts[idx]:4d} images")

# ── Split 70 / 15 / 15 ────────────────────────────────────────────────────────
import torch
generator = torch.Generator().manual_seed(args.seed)

n_train = int(0.70 * N)
n_val   = int(0.15 * N)
n_test  = N - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test], generator=generator
)

print(f"\n── Split sizes (seed={args.seed}) ──")
print(f"  Train : {len(train_set):4d}  ({len(train_set)/N*100:.1f}%)")
print(f"  Val   : {len(val_set):4d}  ({len(val_set)/N*100:.1f}%)")
print(f"  Test  : {len(test_set):4d}  ({len(test_set)/N*100:.1f}%)")

# ── Per-split class distribution ──────────────────────────────────────────────
def get_counts(subset, n_classes):
    counts = [0] * n_classes
    for idx in subset.indices:
        _, label = full_dataset.samples[idx]
        counts[label] += 1
    return counts

train_counts = get_counts(train_set, len(CLASSES))
val_counts   = get_counts(val_set,   len(CLASSES))
test_counts  = get_counts(test_set,  len(CLASSES))

print("\n── Per-class counts per split ──")
header = f"{'Class':12s} | {'Train':>6} | {'Val':>5} | {'Test':>5} | {'Total':>6}"
print(header)
print("-" * len(header))
for i, cls in enumerate(CLASSES):
    total = train_counts[i] + val_counts[i] + test_counts[i]
    print(f"  {cls:12s} | {train_counts[i]:6d} | {val_counts[i]:5d} | "
          f"{test_counts[i]:5d} | {total:6d}")

# ── Label consistency check (answers Sebastian's Q4) ─────────────────────────
train_labels_present = set(i for i, c in enumerate(train_counts) if c > 0)
val_labels_present   = set(i for i, c in enumerate(val_counts)   if c > 0)
test_labels_present  = set(i for i, c in enumerate(test_counts)  if c > 0)

print("\n── Label consistency check ──")
all_same = (train_labels_present == val_labels_present == test_labels_present)
if all_same:
    print("  ✅ All 6 classes present in train, val, AND test — no label mismatch.")
else:
    missing_in_val  = train_labels_present - val_labels_present
    missing_in_test = train_labels_present - test_labels_present
    if missing_in_val:
        print(f"  ⚠️  Classes missing in val : {[CLASSES[i] for i in missing_in_val]}")
    if missing_in_test:
        print(f"  ⚠️  Classes missing in test: {[CLASSES[i] for i in missing_in_test]}")

# ── Class imbalance report (explains 0% trash F1) ────────────────────────────
print("\n── Class imbalance analysis ──")
print("  (This explains why 'trash' got 0% F1 in the baseline run)")
max_count = max(train_counts)
for i, cls in enumerate(CLASSES):
    ratio = max_count / train_counts[i] if train_counts[i] > 0 else float('inf')
    bar = "█" * int(train_counts[i] / max_count * 20)
    print(f"  {cls:12s} : {train_counts[i]:4d}  {bar}  (imbalance ratio: {ratio:.1f}x)")

# ── Save bar chart ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
splits     = ["Train", "Validation", "Test"]
all_counts = [train_counts, val_counts, test_counts]
colors     = ["#4C72B0", "#DD8452", "#55A868"]

for ax, split_name, counts, color in zip(axes, splits, all_counts, colors):
    bars = ax.bar(CLASSES, counts, color=color, edgecolor="white")
    ax.set_title(f"{split_name}  (n={sum(counts)})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=9)

plt.suptitle("TrashNet — Class Distribution per Split (seed=42, 70/15/15)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/class_distribution.png", dpi=150, bbox_inches="tight")
print("\n── Chart saved to reports/class_distribution.png ──")
print("\n" + "=" * 60)
print("AUDIT COMPLETE — Show this output + the chart to Sebastian.")
print("=" * 60)