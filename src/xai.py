"""
xai.py - GradCAM visualizations for waste classification models
Generates heatmaps showing what the model focuses on when classifying images.
Used to explain correct predictions and analyze failure cases on OOD data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform

from models import get_model

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

REALWASTE_MAP = {
    'Cardboard': 'cardboard', 'Glass': 'glass', 'Metal': 'metal',
    'Paper': 'paper', 'Plastic': 'plastic', 'Miscellaneous Trash': 'trash',
    'Food Organics': 'trash', 'Textile Trash': 'trash', 'Vegetation': 'trash',
}


def get_target_layer(model, model_name):
    if model_name == 'resnet50':
        return [model.layer4[-1]]
    elif model_name == 'efficientnet':
        return [model.conv_head]
    elif model_name == 'vit':
        return [model.blocks[-1].norm1]


def load_realwaste_samples(realwaste_root):
    samples = []
    for folder_name, unified_name in REALWASTE_MAP.items():
        class_dir = Path(realwaste_root) / folder_name
        if not class_dir.exists():
            continue
        label_idx = CLASS_TO_IDX[unified_name]
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                samples.append((str(img_file), label_idx))
    return samples


def generate_gradcam(model_name, checkpoint_path, realwaste_root,
                     output_dir='reports', n_samples=6):
    """
    Generate GradCAM visualizations for correct and incorrect predictions.

    Args:
        model_name: 'resnet50', 'efficientnet', or 'vit'
        checkpoint_path: path to .pth checkpoint
        realwaste_root: path to RealWaste dataset root
        output_dir: where to save the output image
        n_samples: number of correct/wrong examples to show
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating GradCAM for {model_name} on {device}...")

    model = get_model(model_name, num_classes=6)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    target_layers = get_target_layer(model, model_name)
    reshape = vit_reshape_transform if model_name == 'vit' else None
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape)

    samples = load_realwaste_samples(realwaste_root)
    correct_samples, wrong_samples = [], []

    for img_path, true_label in samples:
        if len(correct_samples) >= n_samples and len(wrong_samples) >= n_samples:
            break
        image = Image.open(img_path).convert('RGB')
        tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).argmax(1).item()
        if pred == true_label and len(correct_samples) < n_samples:
            correct_samples.append((true_label, pred, tensor))
        elif pred != true_label and len(wrong_samples) < n_samples:
            wrong_samples.append((true_label, pred, tensor))

    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 4, 16))
    fig.suptitle(
        f'GradCAM — {model_name} on RealWaste (OOD)\n'
        f'Rows 1-2: Correct predictions | Rows 3-4: Wrong predictions',
        fontsize=13, fontweight='bold'
    )

    for col_idx, (true_label, pred, tensor) in enumerate(correct_samples):
        grayscale_cam = cam(input_tensor=tensor,
                            targets=[ClassifierOutputTarget(pred)])[0]
        rgb_img = np.clip(
            INV_NORMALIZE(tensor.squeeze()).permute(1, 2, 0).cpu().numpy(), 0, 1)
        viz = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axes[0, col_idx].imshow(rgb_img)
        axes[0, col_idx].set_title(f'True: {CLASSES[true_label]}',
                                    fontsize=9, color='green')
        axes[0, col_idx].axis('off')
        axes[1, col_idx].imshow(viz)
        axes[1, col_idx].set_title(f'Pred: {CLASSES[pred]} ✓',
                                    fontsize=9, color='green')
        axes[1, col_idx].axis('off')

    for col_idx, (true_label, pred, tensor) in enumerate(wrong_samples):
        grayscale_cam = cam(input_tensor=tensor,
                            targets=[ClassifierOutputTarget(pred)])[0]
        rgb_img = np.clip(
            INV_NORMALIZE(tensor.squeeze()).permute(1, 2, 0).cpu().numpy(), 0, 1)
        viz = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axes[2, col_idx].imshow(rgb_img)
        axes[2, col_idx].set_title(f'True: {CLASSES[true_label]}',
                                    fontsize=9, color='red')
        axes[2, col_idx].axis('off')
        axes[3, col_idx].imshow(viz)
        axes[3, col_idx].set_title(f'Pred: {CLASSES[pred]} ✗',
                                    fontsize=9, color='red')
        axes[3, col_idx].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'gradcam_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--realwaste', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='reports')
    args = parser.parse_args()

    generate_gradcam(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        realwaste_root=args.realwaste,
        output_dir=args.output_dir,
    )
