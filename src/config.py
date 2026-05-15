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
    },
    'efficientnet': {
        'model_name': 'efficientnet_b3',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 20,
        'optimizer': 'AdamW',
        'status': 'pending',
    },
    'vit': {
        'model_name': 'vit_small_patch16_224',
        'lr': 5e-5,
        'batch_size': 16,
        'epochs': 20,
        'optimizer': 'AdamW',
        'status': 'pending',
    },
}

NUM_CLASSES = 6
UNIFIED_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
RANDOM_SEED = 42
