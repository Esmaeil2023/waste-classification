"""
experiment_log.py - Experiment tracking and results log
Author: khawar khan

This file tracks all experiments run in this project.
Update this file every time you run a new experiment.
"""

EXPERIMENT_RESULTS = [
    {
        'id': 'EXP001',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet',
        'dataset_test': 'TrashNet (test split)',
        'epochs': 5,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.672,
        'status': 'DONE',
        'notes': 'Baseline experiment. Trash class = 0% (only 137 training images)',
    },
    {
        'id': 'EXP002',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet',
        'dataset_test': 'RealWaste (OOD)',
        'epochs': 5,
        'balanced_accuracy': 0.378,
        'domain_gap': -0.294,
        'status': 'DONE',
        'notes': 'Core finding: 29.4% accuracy drop. Paper recall=0.98 but precision=0.25',
    },
    {
        'id': 'EXP003b',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet',
        'dataset_test': 'TrashNet (test split)',
        'epochs': 5,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.745,
        'status': 'DONE',
        'notes': 'Class-weighted loss fix. Trash F1: 0% -> 46.2%. Overall: 67.2% -> 74.5%',
    },
    {
        'id': 'EXP004b',
        'model': 'EfficientNet-B3',
        'dataset_train': 'TrashNet',
        'dataset_test': 'TrashNet (test split)',
        'epochs': 20,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.880,
        'status': 'DONE',
        'notes': 'Weighted loss. Trash F1: 80%. Significant improvement over ResNet.',
    },
    {
        'id': 'EXP005',
        'model': 'ViT-Small/16',
        'dataset_train': 'TrashNet',
        'dataset_test': 'TrashNet (test split)',
        'epochs': 20,
        'optimizer': 'AdamW',
        'lr': 5e-5,
        'balanced_accuracy': 0.942,
        'status': 'DONE',
        'notes': 'Best model. Trash F1: 87%. ViT outperforms CNNs on this dataset.',
    },
    {
        'id': 'EXP006',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet + GD (18042 images)',
        'dataset_test': 'Combined test split',
        'epochs': 20,
        'balanced_accuracy': 0.955,
        'status': 'DONE',
        'notes': 'Combined dataset. Massive improvement from 74.5% to 95.5%.',
    },
    {
        'id': 'EXP007',
        'model': 'EfficientNet-B3',
        'dataset_train': 'TrashNet + GD (18042 images)',
        'dataset_test': 'Combined test split',
        'epochs': 20,
        'balanced_accuracy': 0.960,
        'status': 'DONE',
        'notes': 'Combined dataset. 96.0% balanced accuracy.',
    },
    {
        'id': 'EXP008',
        'model': 'ViT-Small/16',
        'dataset_train': 'TrashNet + GD (18042 images)',
        'dataset_test': 'Combined test split',
        'epochs': 20,
        'balanced_accuracy': 0.974,
        'status': 'DONE',
        'notes': 'Best model. 97.4% balanced accuracy. All classes above 94% F1.',
    },
    {
        'id': 'EXP009',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet + GD',
        'dataset_test': 'RealWaste (OOD)',
        'balanced_accuracy': 0.600,
        'domain_gap': -0.355,
        'status': 'DONE',
        'notes': 'OOD eval. 35.5% drop. Cardboard hardest class (F1=0.49).',
    },
    {
        'id': 'EXP010',
        'model': 'EfficientNet-B3',
        'dataset_train': 'TrashNet + GD',
        'dataset_test': 'RealWaste (OOD)',
        'balanced_accuracy': 0.536,
        'domain_gap': -0.424,
        'status': 'DONE',
        'notes': 'Worst OOD. 42.4% drop. Overfit to clean backgrounds.',
    },
    {
        'id': 'EXP011',
        'model': 'ViT-Small/16',
        'dataset_train': 'TrashNet + GD',
        'dataset_test': 'RealWaste (OOD)',
        'balanced_accuracy': 0.622,
        'domain_gap': -0.352,
        'status': 'DONE',
        'notes': 'Best OOD. ViT attention mechanism generalizes better than CNNs.',
    },
    {
        'id': 'EXP012',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet + GD (albumentations augmentation)',
        'dataset_test': 'Combined test split',
        'epochs': 20,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.974,
        'status': 'DONE',
        'notes': 'Mitigation attempt #1: replaced torchvision aug with albumentations '
                 '(motion blur, gaussian blur, shadows, brightness/contrast, gauss noise, '
                 'coarse dropout). In-dist accuracy improved slightly: 95.5% -> 97.4%.',
    },
    {
        'id': 'EXP013',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet + GD (albumentations augmentation)',
        'dataset_test': 'RealWaste (OOD)',
        'balanced_accuracy': 0.570,
        'domain_gap': -0.404,
        'status': 'DONE',
        'notes': 'Mitigation attempt #1 result: OOD accuracy DECREASED 60.0% -> 57.0%, '
                 'domain gap WORSENED -35.5% -> -40.4%. Negative result: heavier synthetic '
                 'augmentation on the SAME limited training images increased in-distribution '
                 'overconfidence without improving real-world transfer. Suggests the gap is '
                 'driven more by lack of training image diversity than by lack of synthetic '
                 'noise/blur/occlusion. Next: test a third, visually diverse training dataset '
                 '(sumn2u/garbage-classification-v2) instead of synthetic augmentation alone.',
    },
    {
        'id': 'EXP014',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet + GD + sumn2u (30,301 images, no heavy aug)',
        'dataset_test': '3-dataset combined test split',
        'epochs': 20,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.974,
        'status': 'DONE',
        'notes': 'Mitigation attempt #2: added a THIRD training dataset '
                 '(sumn2u/garbage-classification-v2, 12,259 images, 10 classes mapped to our 6) '
                 'on top of TrashNet+GD, using plain (non-augmented) transforms to isolate the '
                 'effect of dataset diversity alone. In-dist accuracy: 97.4%, all classes >=95% F1.',
    },
    {
        'id': 'EXP015',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet + GD + sumn2u (30,301 images, no heavy aug)',
        'dataset_test': 'RealWaste (OOD)',
        'balanced_accuracy': 0.564,
        'domain_gap': -0.410,
        'status': 'DONE',
        'notes': 'Mitigation attempt #2 result: OOD accuracy DECREASED further to 56.4% '
                 '(worst of all 3 ResNet-50 OOD runs so far). Domain gap WORSENED to -41.0%, '
                 'the largest gap recorded for ResNet-50. KEY FINDING: both mitigation attempts '
                 '(synthetic augmentation in EXP012/013, and adding a 3rd same-domain dataset '
                 'here) consistently INCREASE in-distribution accuracy while DECREASING OOD '
                 'accuracy, in that order: 95.5/60.0 -> 97.4/57.0 -> 97.4/56.4. This indicates '
                 'the domain gap is not caused by insufficient training data volume or '
                 'diversity WITHIN similar-style sources (TrashNet, GD12, and sumn2u are all '
                 'clean-ish single-object photography despite different backgrounds/cameras) - '
                 'it is a genuine domain mismatch with RealWaste (landfill conditions: '
                 'deformed/crushed objects, dirt, multi-object frames, harsh lighting). '
                 'Adding more of the same KIND of data lets the model memorize more '
                 'source-specific surface cues (background, print/logos) without learning '
                 'anything that transfers. Recommendation: the own/hard team dataset '
                 '(real-world conditions) is likely a more promising mitigation lever than '
                 'further augmentation or same-domain dataset expansion.',
    },
    {
        'id': 'EXP003',
        'model': 'EfficientNet-B3',
        'dataset_train': 'TrashNet',
        'dataset_test': 'TrashNet + RealWaste',
        'epochs': 20,
        'balanced_accuracy': None,
        'status': 'PENDING',
        'notes': 'Needs GPU - planned for Google Colab',
    },
    {
        'id': 'EXP004',
        'model': 'ViT-Small/16',
        'dataset_train': 'TrashNet',
        'dataset_test': 'TrashNet + RealWaste',
        'epochs': 20,
        'balanced_accuracy': None,
        'status': 'PENDING',
        'notes': 'Attention-based model - hypothesis: better OOD generalization',
    },
]