"""
evaluation_plan.py - Evaluation strategy and XAI plan
Author: Khaled Ibrahim

This file documents our evaluation methodology and
explainability analysis plan.
"""
import os

from torch.cuda import device

import models
from datasets import TrashNetDataset, get_trashnet_loaders
from evaluate import load_model_from_checkpoint

EVALUATION_METRICS = {
    'primary': 'balanced_accuracy',
    'reason': 'Classes are imbalanced (trash only has 137 images). '
              'Balanced accuracy treats all classes equally.',
    'secondary': ['f1_per_class', 'precision', 'recall'],
    'planned': ['AUROC', 'ECE (calibration error)'],
}

OOD_DOMAINS = [
    {
        'name': 'RealWaste',
        'status': 'DONE',
        'result': '37.8% balanced accuracy (vs 67.2% in-distribution)',
        'domain_gap': '-29.4%',
    },
    {
        'name': 'TACO',
        'status': 'PENDING',
        'result': None,
    },
    {
        'name': 'Own team dataset',
        'status': 'PLANNED (weeks 5-10)',
        'result': None,
    },
]

XAI_PLAN = {
    'GradCAM': {
        'purpose': 'Show which pixels the model focuses on per class',
        'status': 'IN PROGRESS - see xai.py',
        'hypothesis': 'Model focuses on background not material texture',
    },
    'LIME': {
        'purpose': 'Show which image regions drive the prediction',
        'status': 'PLANNED',
    },
}

PER_CLASS_OOD_RESULTS = {
    'cardboard': {'indist_f1': 0.811, 'ood_f1': 0.054, 'drop': '94%'},
    'glass':     {'indist_f1': 0.811, 'ood_f1': 0.326, 'drop': '60%'},
    'metal':     {'indist_f1': 0.797, 'ood_f1': 0.601, 'drop': '25%'},
    'paper':     {'indist_f1': 0.804, 'ood_f1': 0.398, 'drop': '50%'},
    'plastic':   {'indist_f1': 0.746, 'ood_f1': 0.511, 'drop': '31%'},
    'trash':     {'indist_f1': 0.000, 'ood_f1': 0.000, 'drop': 'N/A'},
}

import torch
import numpy as np
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries


class LimeExplainer:
    """
    Clean LIME wrapper for PyTorch image classification models
    """

    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()

        self.explainer = lime_image.LimeImageExplainer()

    # ---------------------------------------------
    # 1. Prediction function (IMPORTANT for LIME)
    # ---------------------------------------------
    def _predict_proba(self, images):
        """
        images: numpy array (N, H, W, C) in [0..255]
        returns: probabilities (N, num_classes)
        """

        self.model.eval()
        batch = []

        for img in images:
            img = torch.tensor(img).float()

            # HWC -> CHW
            img = img.permute(2, 0, 1)

            # Normalize (adjust if you used different normalization)
            img = img / 255.0

            batch.append(img)

        batch = torch.stack(batch).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    # ---------------------------------------------
    # 2. Explain single image
    # ---------------------------------------------
    def explain(self, image, top_labels=1, num_samples=1000):
        """
        image: numpy array (H, W, C) in uint8
        """

        explanation = self.explainer.explain_instance(
            image,
            self._predict_proba,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )

        label = explanation.top_labels[0]

        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        return temp, mask, label

    # ---------------------------------------------
    # 3. Visualization
    # ---------------------------------------------
    def plot(self, image, temp, mask, label):
        plt.figure(figsize=(10, 5))

        # Original + explanation overlay
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"LIME Explanation: {self.class_names[label]}")

        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

# class names
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
model = models.get_model("resnet50", num_classes=6)
load_model_from_checkpoint("/content/drive/MyDrive/ColabNotebooks/resnet50_best.pth",
                           "resnet50",
                                6, device)


lime_explainer = LimeExplainer(model, device, class_names)
data_set = TrashNetDataset(os.path.expanduser('/content/drive/MyDrive/ColabNotebooks/data/raw/Trashnet'))

train_loader, val_loader, test_loader = get_trashnet_loaders('/content/drive/MyDrive/ColabNotebooks/data/raw/Trashnet')  # numpy (H, W, C)
image = test_loader.dataset[1]
temp, mask, label = lime_explainer.explain(image)

lime_explainer.plot(image, temp, mask, label)