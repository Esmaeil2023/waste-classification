# Waste Classification — Domain Generalization Study

Master's project for xAI-Proj-M, Chair of Explainable Machine Learning, University of Bamberg.
Supervisor: Sebastian Doerrich.

## Project Goal

Train image classifiers to sort waste into 6 categories (cardboard, glass, metal,
paper, plastic, trash), then measure how well they generalize from clean training
data to real-world, out-of-distribution (OOD) images.

## Datasets

**Training (combined, 18,042 images):**
- TrashNet — 2,527 images, clean white background
- Garbage Classification (12-class) — 15,515 images, mapped to our 6 classes

**OOD Testing:**
- RealWaste — 4,752 images of real landfill waste, never seen during training

## Models

Three architectures trained for 20 epochs each (AdamW, batch size 32):

| Model | In-Distribution Acc | OOD Acc (RealWaste) | Domain Gap |
|---|---|---|---|
| ResNet-50 | 95.5% | 60.0% | -35.5% |
| EfficientNet-B3 | 96.0% | 53.6% | -42.4% |
| ViT-Small/16 | 97.4% | 62.2% | -35.2% |

**Key finding:** all models lose 35-42% accuracy when tested on real-world images.
ViT generalizes best despite CNNs achieving similar in-distribution accuracy —
its attention mechanism appears more robust to domain shift. EfficientNet overfits
most to clean backgrounds.

## Explainability (GradCAM)

GradCAM visualizations for all 3 models on RealWaste OOD samples are in `reports/`.
They show which image regions each model focuses on for correct vs incorrect
predictions, helping explain why the domain gap occurs (e.g. focusing on
background or unrelated print/logos rather than material texture).

## Project Structure

src/

datasets.py          - TrashNet and GD dataset loaders, class mapping

models.py             - ResNet-50, EfficientNet-B3, ViT-Small builders

train.py               - training loop with weighted loss for class imbalance

evaluate.py            - OOD evaluation on RealWaste

xai.py                  - GradCAM generation for all 3 models (handles ViT separately)

metrics.py             - balanced accuracy, domain gap computation

config.py              - experiment configuration and results summary

data_audit.py          - verifies train/val/test split and label consistency

experiment_log.py      - full experiment history (EXP001-EXP011)

evaluation_plan.py     - evaluation methodology documentation

reports/

class_distribution.png    - TrashNet split visualization

gradcam_resnet50.png       - GradCAM for ResNet-50

gradcam_efficientnet.png   - GradCAM for EfficientNet-B3

gradcam_vit.png             - GradCAM for ViT-Small## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Data audit (verify split and class balance)
```bash
python src/data_audit.py --data_dir data/raw/trashnet
```

### Train a model
```bash
python src/train.py
```
Edit the CONFIG dict at the top of train.py to change model, epochs, or hyperparameters.

### OOD evaluation
```bash
python src/evaluate.py
```
Update CHECKPOINT_PATH and REALWASTE_ROOT at the bottom of the file to point
to your trained checkpoint and dataset location.

### Generate GradCAM visualizations
```bash
python src/xai.py --model resnet50 --checkpoint path/to/checkpoint.pth --realwaste path/to/RealWaste
```

## Team

- Esmaeil Molapour — data pipeline, model training, OOD evaluation, GradCAM/XAI
- Khaled Ibrahim — evaluation planning, AUROC/ECE metrics
- Khawar Khan — experiment tracking
- Ashley — iterative improvement strategy (planned)
