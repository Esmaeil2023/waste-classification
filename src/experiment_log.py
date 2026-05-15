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
