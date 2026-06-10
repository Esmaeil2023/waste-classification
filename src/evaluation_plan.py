"""
evaluation_plan.py - Evaluation strategy and XAI plan
Author: Khaled Ibrahim

This file documents our evaluation methodology and
explainability analysis plan.
"""

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
