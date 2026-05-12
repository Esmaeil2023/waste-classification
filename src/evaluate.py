import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Callable
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report
import numpy as np

# Import our modules — note: no cv2 imported here to avoid SIGSEGV
from models import get_model

# ── What this file does ──────────────────────────────────────────────────────
# Loads a trained model checkpoint and evaluates it on OOD (out-of-distribution)
# datasets. This is the core of our domain generalization research:
#
#   - Train on TrashNet (clean, white background)
#   - Test on RealWaste (real landfill images) → measure the accuracy DROP
#
# The drop between in-distribution and OOD accuracy is the key research finding.
# A small drop = good domain generalization. A large drop = model overfit to
# the source domain.
# ─────────────────────────────────────────────────────────────────────────────

# ── Unified class mapping ─────────────────────────────────────────────────────
# Our model was trained on 6 TrashNet classes.
# RealWaste has 9 classes — we map them to our 6.
# Classes that don't map (Food Organics, Textile Trash, Vegetation) are skipped
# during evaluation since our model was never trained on them.
UNIFIED_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(UNIFIED_CLASSES)}

# RealWaste folder names → our unified class names
# Food Organics, Textile Trash, Vegetation → None (skip these)
REALWASTE_MAP = {
    'Cardboard':           'cardboard',
    'Glass':               'glass',
    'Metal':               'metal',
    'Paper':               'paper',
    'Plastic':             'plastic',
    'Miscellaneous Trash': 'trash',
    'Food Organics':       None,   # no equivalent in TrashNet → skip
    'Textile Trash':       None,   # no equivalent in TrashNet → skip
    'Vegetation':          None,   # no equivalent in TrashNet → skip
}
# ─────────────────────────────────────────────────────────────────────────────


def get_eval_transform():
    """Standard evaluation transform — same as val/test in train.py."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class RealWasteDataset(Dataset):
    """
    Loads RealWaste dataset and maps its 9 classes to our 6 unified classes.
    Images from unmapped classes (Food Organics, Textile Trash, Vegetation)
    are skipped — our model was never trained to predict these.

    Structure expected:
        root/
            Cardboard/
            Glass/
            Metal/
            Paper/
            Plastic/
            Miscellaneous Trash/
            Food Organics/      ← skipped
            Textile Trash/      ← skipped
            Vegetation/         ← skipped
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # list of (image_path, unified_label_index)
        self.skipped_classes = []

        for folder_name, unified_name in REALWASTE_MAP.items():
            class_dir = self.root / folder_name
            if not class_dir.exists():
                print(f"  Warning: folder not found: {class_dir}")
                continue

            if unified_name is None:
                # Count skipped images for reporting
                n = len(list(class_dir.glob('*.jpg')))
                self.skipped_classes.append(f"{folder_name} ({n} images)")
                continue

            label_idx = CLASS_TO_IDX[unified_name]
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), label_idx))

        print(f"RealWasteDataset loaded: {len(self.samples)} images")
        if self.skipped_classes:
            print(f"  Skipped classes (not in TrashNet): {', '.join(self.skipped_classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


@torch.no_grad()
def evaluate_on_dataset(model, loader, criterion, device, dataset_name: str) -> dict:
    """
    Evaluates a model on a single dataset and prints results.

    Args:
        model: trained PyTorch model
        loader: DataLoader for the dataset
        criterion: loss function
        device: cpu or cuda or mps
        dataset_name: just for printing (e.g. 'RealWaste OOD')

    Returns:
        dict with loss, balanced_acc, and per-class report
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"\nEvaluating on: {dataset_name}")
    print(f"  Total batches: {len(loader)}")

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}")

    avg_loss = total_loss / len(loader)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Get the actual class names present in this dataset
    present_classes = sorted(set(all_labels))
    present_class_names = [UNIFIED_CLASSES[i] for i in present_classes]

    report = classification_report(
        all_labels, all_preds,
        labels=present_classes,
        target_names=present_class_names,
        output_dict=True,
        zero_division=0
    )

    print(f"\n  Results for {dataset_name}:")
    print(f"  Loss:             {avg_loss:.4f}")
    print(f"  Balanced Acc:     {bal_acc:.4f} ({bal_acc*100:.1f}%)")
    print(f"\n  Per-class breakdown:")
    for cls in present_class_names:
        r = report[cls]
        print(f"    {cls:22s} precision: {r['precision']:.3f} | "
              f"recall: {r['recall']:.3f} | f1: {r['f1-score']:.3f}")

    return {
        'dataset': dataset_name,
        'loss': avg_loss,
        'balanced_acc': bal_acc,
        'report': report,
    }


def load_model_from_checkpoint(checkpoint_path: str, model_name: str,
                                num_classes: int, device) -> nn.Module:
    """
    Loads a trained model from a .pth checkpoint file.

    The checkpoint was saved by train.py and contains:
        - model_state_dict: the trained weights
        - optimizer_state_dict: optimizer state (not needed for eval)
        - val_loss: best validation loss
        - config: the training config dict
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Checkpoint epoch:    {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val loss at save:    {checkpoint.get('val_loss', 'unknown'):.4f}")
    print(f"  Val bal.acc at save: {checkpoint.get('val_balanced_acc', 'unknown'):.4f}")

    return model


if __name__ == '__main__':
    # ── Configuration ──────────────────────────────────────────────────────
    CHECKPOINT_PATH = os.path.expanduser(
        '~/waste-classification/experiments/resnet50_best.pth'
    )
    MODEL_NAME   = 'resnet50'
    NUM_CLASSES  = 6
    BATCH_SIZE   = 32

    REALWASTE_ROOT = os.path.expanduser(
        '~/waste-classification/data/raw/realwaste/realwaste-main/RealWaste'
    )
    # ───────────────────────────────────────────────────────────────────────

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(
        CHECKPOINT_PATH, MODEL_NAME, NUM_CLASSES, device
    )

    criterion = nn.CrossEntropyLoss()
    transform = get_eval_transform()

    # ── Evaluate on RealWaste (OOD) ────────────────────────────────────────
    realwaste_dataset = RealWasteDataset(REALWASTE_ROOT, transform=transform)
    realwaste_loader = DataLoader(
        realwaste_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    ood_results = evaluate_on_dataset(
        model, realwaste_loader, criterion, device,
        dataset_name='RealWaste (OOD)'
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DOMAIN GENERALIZATION SUMMARY")
    print("=" * 60)
    print(f"  Train domain:  TrashNet (clean, white background)")
    print(f"  Test domain:   RealWaste (real landfill)")
    print(f"  In-dist acc:   67.2%  (from train.py, 5 epochs)")
    print(f"  OOD acc:       {ood_results['balanced_acc']*100:.1f}%")
    drop = 67.2 - ood_results['balanced_acc'] * 100
    print(f"  Domain gap:    {drop:.1f}% drop")
    print("=" * 60)
    print("\nThis domain gap is the core finding of your project.")
    print("A large gap motivates domain generalization techniques.")