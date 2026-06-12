"""
config.py - Experiment configuration
Author: Esmaeil Molapour
"""

CONFIGS = {
    'resnet50': {
        'model_name': 'resnet50',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 20,
        'optimizer': 'AdamW',
        'status': 'done',
        'in_dist_acc': 0.955,
        'ood_acc': 0.600,
    },
    'efficientnet': {
        'model_name': 'efficientnet_b3',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 20,
        'optimizer': 'AdamW',
        'status': 'done',
        'in_dist_acc': 0.960,
        'ood_acc': 0.536,
    },
    'vit': {
        'model_name': 'vit_small_patch16_224',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 20,
        'optimizer': 'AdamW',
        'status': 'done',
        'in_dist_acc': 0.974,
        'ood_acc': 0.622,
    },
}

DATASETS = {
    'train': ['TrashNet', 'GarbageClassification12'],
    'ood_test': ['RealWaste'],
    'own_test': 'TBD',
}

NUM_CLASSES = 6
UNIFIED_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15