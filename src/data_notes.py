"""
data_notes.py - Dataset documentation and analysis notes
Author: Ashly

Datasets in this project:
- TrashNet: 2527 images, 6 classes, clean white background ? used for TRAINING
- RealWaste: 4752 images, 9 classes, real landfill photos ? used for OOD TEST
- TACO: ~800 images, outdoor litter ? used for OOD TEST

Class mapping (RealWaste to TrashNet):
- Cardboard        ? cardboard
- Glass            ? glass
- Metal            ? metal
- Paper            ? paper
- Plastic          ? plastic
- Miscellaneous    ? trash
- Food Organics    ? SKIPPED (no equivalent in TrashNet)
- Textile Trash    ? SKIPPED (no equivalent in TrashNet)
- Vegetation       ? SKIPPED (no equivalent in TrashNet)

Key finding: model predicts paper for almost everything in RealWaste
? recall 98% but precision only 25%
? model learned background color, not material texture
"""

DATASET_STATS = {
    'trashnet': {
        'total_images': 2527,
        'classes': 6,
        'split': '70% train / 15% val / 15% test',
        'background': 'white, controlled studio',
        'role': 'training domain',
    },
    'realwaste': {
        'total_images': 4752,
        'classes': 9,
        'mapped_classes': 6,
        'skipped_classes': 3,
        'background': 'real landfill, mixed',
        'role': 'OOD test domain',
    },
    'taco': {
        'total_images': 800,
        'classes': 'multiple',
        'background': 'outdoor, varying',
        'role': 'OOD test domain (pending)',
    },
}
