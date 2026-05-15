"""
metrics.py - Evaluation metrics and domain gap computation
Author: Esmaeil Molapour
"""
from sklearn.metrics import balanced_accuracy_score, classification_report

def compute_metrics(all_labels, all_preds, class_names):
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    return {
        'balanced_accuracy': bal_acc,
        'per_class': {cls: report[cls] for cls in class_names},
    }

def compute_domain_gap(indist_acc, ood_acc):
    gap = indist_acc - ood_acc
    print(f"In-distribution: {indist_acc*100:.1f}%")
    print(f"OOD:             {ood_acc*100:.1f}%")
    print(f"Domain gap:      -{gap*100:.1f}%")
    return gap
