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
                 'noise/blur/occlusion.',
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
                 '(worst of all 3 ResNet-50 OOD runs at the time). Domain gap WORSENED to -41.0%. '
                 'KEY FINDING: both mitigation attempts so far (synthetic augmentation in '
                 'EXP012/013, and adding a 3rd same-domain dataset here) consistently INCREASE '
                 'in-distribution accuracy while DECREASING OOD accuracy: 95.5/60.0 -> '
                 '97.4/57.0 -> 97.4/56.4. TrashNet, GD12, and sumn2u are all clean-ish '
                 'single-object photography despite different backgrounds/cameras - this is '
                 'a genuine domain mismatch with RealWaste (landfill conditions: '
                 'deformed/crushed objects, dirt, multi-object frames, harsh lighting), not a '
                 'lack of training data volume or stylistic diversity. sumn2u DROPPED from '
                 'further experiments based on this result. Recommendation: the own/hard team '
                 'dataset or limited exposure to real OOD data is a more promising lever.',
    },
    {
        'id': 'EXP016',
        'model': 'ResNet-50 / EfficientNet-B3 / ViT-Small',
        'dataset_train': 'TrashNet + GD (no retraining - diagnostic only)',
        'dataset_test': 'RealWaste (OOD), trash class excluded at eval time',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'Diagnostic: re-evaluated existing combined checkpoints on RealWaste with '
                 'the trash class removed from scoring, to test whether trash label '
                 'heterogeneity (battery/biological/clothes/shoes/food/vegetation all merged) '
                 'was dragging down the OOD average. RESULT WAS OPPOSITE OF HYPOTHESIS: removing '
                 'trash DECREASED balanced accuracy for all 3 models (ResNet -6.4pts, '
                 'EfficientNet -7.7pts, ViT -2.5pts). Trash is actually the highest-recall class '
                 'on RealWaste (majority class, ~35% of samples), so removing it from the '
                 'macro-average removes the model\'s best class, leaving only the 5 harder '
                 'classes. CONCLUSION: does not support label heterogeneity in "trash" as the '
                 'primary cause of the domain gap; more consistent with surface-level feature '
                 'reliance (background/print, per GradCAM evidence) than with noisy trash '
                 'labels specifically. NOTE: 6-class baseline reproduced here as 54.7% for '
                 'ResNet-50 in one run vs originally logged 60.0% (EXP009) in another; '
                 'flagged for re-verification.',
    },
    {
        'id': 'EXP017',
        'model': 'ResNet-50',
        'dataset_train': 'TrashNet + GD + 15% of RealWaste (mitigation: partial domain exposure)',
        'dataset_test': 'RealWaste, remaining held-out 85% (never seen in training)',
        'epochs': 20,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.764,
        'status': 'DONE',
        'notes': 'BEST MITIGATION RESULT (mitigation attempt #3). Mixed a small slice (15%, '
                 '712 images) of RealWaste into training alongside TrashNet+GD, keeping the '
                 'remaining 85% (4,040 images) fully held out as the OOD test set. Balanced '
                 'accuracy on held-out RealWaste jumped from 60.0% (zero-shot baseline, EXP009) '
                 'to 76.4% - a +16.4 point improvement, the largest of any mitigation strategy '
                 'tested (vs -3.0pts for augmentation in EXP013, -3.6pts for sumn2u in EXP015). '
                 'Per-class F1 on held-out RealWaste: cardboard 0.70, glass 0.75, metal 0.82, '
                 'paper 0.74, plastic 0.76, trash 0.88. FRAMING NOTE: this changes the research '
                 'question from pure zero-shot domain generalization to limited-supervision '
                 'domain adaptation - the model has now seen a small amount of real target-'
                 'domain data during training. Confirms that exposure to even a little '
                 'genuinely different (real-world) data is far more effective than synthetic '
                 'augmentation or more same-domain data.',
    },
    {
        'id': 'EXP018',
        'model': 'EfficientNet-B3',
        'dataset_train': 'TrashNet + GD + 15% of RealWaste (mitigation: partial domain exposure)',
        'dataset_test': 'RealWaste, remaining held-out 85% (never seen in training)',
        'epochs': 20,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.799,
        'status': 'DONE',
        'notes': 'Same 15% RealWaste fine-tune recipe as EXP017, applied to EfficientNet-B3. '
                 'Balanced accuracy on held-out RealWaste: 79.9%, up from zero-shot 53.6% '
                 '(EXP010) - a +26.3 point improvement, the LARGEST improvement of all 3 '
                 'models. EfficientNet was the worst zero-shot generalizer (most reliant on '
                 'source-domain surface cues), so it had the most to gain from even limited '
                 'real-domain exposure. Per-class F1 on held-out RealWaste: cardboard 0.73, '
                 'glass 0.81, metal 0.81, paper 0.78, plastic 0.78, trash 0.90. Confirms the '
                 'fine-tuning effect generalizes across CNN architectures, not just ResNet-50.',
    },
    {
        'id': 'EXP019',
        'model': 'ViT-Small/16',
        'dataset_train': 'TrashNet + GD + 15% of RealWaste (mitigation: partial domain exposure)',
        'dataset_test': 'RealWaste, remaining held-out 85% (never seen in training)',
        'epochs': 20,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'balanced_accuracy': 0.816,
        'status': 'DONE',
        'notes': 'Same 15% RealWaste fine-tune recipe as EXP017/018, applied to ViT-Small/16. '
                 'Balanced accuracy on held-out RealWaste: 81.6%, up from zero-shot 62.2% '
                 '(EXP011) - a +19.4 point improvement. BEST RESULT OF ALL OOD EVALUATIONS '
                 'in the project (zero-shot and fine-tuned combined). Per-class F1 on held-out '
                 'RealWaste: cardboard 0.76, glass 0.85, metal 0.85, paper 0.80, plastic 0.80, '
                 'trash 0.89. ViT remains the best generalizer even after fine-tuning, '
                 'consistent with EXP011 - attention-based features transfer better than CNN '
                 'features both zero-shot and with limited target-domain supervision. '
                 'SUMMARY ACROSS ALL 3 MODELS (zero-shot -> fine-tuned): ResNet-50 60.0->76.4 '
                 '(+16.4), EfficientNet-B3 53.6->79.9 (+26.3), ViT-Small 62.2->81.6 (+19.4). '
                 'This is the strongest and most consistent mitigation finding in the project.',
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
