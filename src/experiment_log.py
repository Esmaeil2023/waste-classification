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
        'id': 'EXP020',
        'model': 'ResNet-50 / EfficientNet-B3 / ViT-Small (all fine-tuned, EXP017-019)',
        'dataset_train': 'N/A - evaluation only, no retraining',
        'dataset_test': 'TACO (Trash Annotations in Context), 4,784 bounding-box crops',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'Evaluated the 15%-RealWaste-fine-tuned checkpoints (EXP017-019) on TACO, '
                 'a third, fully independent real-world litter dataset never seen in any '
                 'training. TACO is a detection dataset (COCO format, 60 fine categories / 28 '
                 'supercategories); built a category-level map to our 6 classes and cropped '
                 'all 4,784 bounding-box annotations into individual images. RESULT: balanced '
                 'accuracy collapsed to 33.3% (ResNet-50), 34.1% (EfficientNet), 33.8% (ViT) - '
                 'despite the same models scoring 76.4/79.9/81.6% on held-out RealWaste. All '
                 'three architectures converged to within 0.8pts of each other, suggesting a '
                 'shared DATA problem rather than a model-specific generalization failure.',
    },
    {
        'id': 'EXP021',
        'model': 'N/A - data diagnostic',
        'dataset_train': 'N/A',
        'dataset_test': 'TACO crop size analysis',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'Diagnostic: measured crop dimensions for all 4,784 TACO bounding-box crops. '
                 'Found 650/4784 (13.6%) under 32x32px, 1433/4784 (30%) under 64x64px, and '
                 '2024/4784 (42.3%) under 100x100px. Many TACO annotations are small litter '
                 'fragments (cigarette butts, bottle caps, pop tabs) that become low-information '
                 'blobs when upscaled to the model\'s 224x224 input size.',
    },
    {
        'id': 'EXP022',
        'model': 'ResNet-50 / EfficientNet-B3 / ViT-Small (fine-tuned)',
        'dataset_train': 'N/A - evaluation only',
        'dataset_test': 'TACO, filtered to crops >=100x100px (2,760 of 4,784 retained)',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'Re-ran EXP020 after filtering out crops smaller than 100x100px to test '
                 'whether tiny low-information crops explained the collapse. Balanced accuracy '
                 'improved modestly to 38.8% (ResNet-50), 39.0% (EfficientNet), 39.3% (ViT) - '
                 'still far below RealWaste performance. Crop size filtering helped but did not '
                 'resolve the core issue. All models showed the same failure pattern: very low '
                 'precision but very high recall on the trash class (~0.17 precision / ~0.75 '
                 'recall), indicating the models default to predicting "trash" when uncertain '
                 'rather than confidently misclassifying into a specific wrong class.',
    },
    {
        'id': 'EXP023',
        'model': 'ResNet-50 / EfficientNet-B3 / ViT-Small',
        'dataset_train': 'N/A - analysis/conclusion',
        'dataset_test': 'Cross-dataset comparison: RealWaste (held-out) vs TACO (filtered)',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'CONCLUSION of TACO investigation (EXP020-022): the gap is structural, not '
                 'fixable by crop filtering or more fine-tuning. TrashNet/GD12/RealWaste are '
                 'all OBJECT-CENTRIC photographs (single item, centered, fills frame). TACO is '
                 'an IN-CONTEXT LITTER DETECTION dataset (small/partial/fragmented objects in '
                 'natural/urban scenes) - cropping bounding boxes does not convert it into an '
                 'equivalent object-centric classification task. The 15% RealWaste fine-tuning '
                 '(EXP017-019) generalizes WITHIN the object-centric domain (clean studio -> '
                 'real landfill object photos) but does NOT transfer ACROSS the task-structure '
                 'boundary to in-context litter detection. This is now treated as a documented '
                 'scope boundary for the project (see report limitations section), not a '
                 'deficiency in the mitigation strategy. Practical implication for the own/hard '
                 'team dataset: photos must remain object-centric (one item, fills frame, hard '
                 'real-world conditions) to stay within the domain our fine-tuned models can '
                 'address - team given explicit photography protocol to this effect.',
    },
    {
        'id': 'EXP024',
        'model': 'ResNet-50 / EfficientNet-B3 / ViT-Small',
        'dataset_train': 'N/A - evaluation only (zero-shot and fine-tuned checkpoints)',
        'dataset_test': 'Own/team hard dataset, 166-170 images, object-centric protocol',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'First evaluation of own team-collected dataset (own photos, object-centric '
                 'protocol: one item, fills frame, real-world conditions). DIAGNOSTIC FINDING: '
                 'metal class scored exactly 0.00 precision/recall/F1 across ALL 6 model runs '
                 '(3 zero-shot + 3 fine-tuned), consistently. Investigated: no data corruption '
                 '(all images loaded correctly, sensible shapes/pixel ranges). Root cause: '
                 'several metal photos (e.g. tent peg, bottle cap) show a small object against '
                 'a large natural background (grass), similar to TACO\'s failure mode, likely '
                 'due to composition drift during photo collection despite the object-centric '
                 'protocol. Models default to predicting "trash" for ~59% of metal photos '
                 '(10/17), consistent with the "predict majority/uncertain class" behavior also '
                 'seen on TACO. Metal photos flagged for re-shooting with stricter framing.',
    },
    {
        'id': 'EXP025',
        'model': 'ResNet-50 / EfficientNet-B3 / ViT-Small',
        'dataset_train': 'N/A - evaluation only',
        'dataset_test': 'Own/team hard dataset, metal class excluded (5 classes)',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'Re-evaluated own dataset excluding the metal class (see EXP024) to isolate '
                 'a cleaner signal. Zero-shot: ResNet-50 43.6%, EfficientNet-B3 34.6%, '
                 'ViT-Small 53.0% (ViT best, consistent with RealWaste zero-shot ranking). '
                 'Fine-tuned (15% RealWaste): ResNet-50 58.9% (+15.3pts), EfficientNet-B3 '
                 '49.3% (+14.7pts) - both CNNs improved similarly to their RealWaste gains, '
                 'confirming the domain adaptation approach transfers to an independently '
                 'collected real-world dataset for CNN architectures. ViT-Small DECREASED to '
                 '35.0% (-18.0pts) after fine-tuning - the opposite direction from ResNet/'
                 'EfficientNet and from ViT\'s own RealWaste result.',
    },
    {
        'id': 'EXP026',
        'model': 'ViT-Small/16',
        'dataset_train': 'N/A - analysis/conclusion',
        'dataset_test': 'Cross-dataset comparison: RealWaste vs own dataset, zero-shot vs fine-tuned',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'NOTABLE FINDING: ViT-Small was the best zero-shot generalizer on BOTH '
                 'RealWaste (62.2%) and the own dataset (53.0%, excl. metal) - consistent with '
                 'its attention-based architecture generalizing better than CNNs zero-shot '
                 '(matches EXP011). However, after 15%-RealWaste fine-tuning, ViT improved on '
                 'RealWaste (81.6%, EXP019) but WORSENED on the own dataset (35.0%, EXP025) - '
                 'unlike ResNet-50 and EfficientNet-B3, which improved on both. Interpretation: '
                 'ViT\'s attention mechanism may adapt more specifically/narrowly to the exact '
                 'visual style of whatever real-world data it is fine-tuned on (RealWaste\'s '
                 'landfill setting), at the cost of transferring to a DIFFERENT independently-'
                 'collected real-world dataset (own photos: varied backgrounds like grass, '
                 'wood floors, concrete). This suggests domain adaptation benefit is not just '
                 'architecture-dependent but also may not generalize uniformly across distinct '
                 'real-world sub-domains, even for the same "real-world" super-category. '
                 'Genuine, reportable finding - not chased further as a bug.',
    },
    {
        'id': 'EXP027',
        'model': 'N/A - methodology/framing decision',
        'dataset_train': 'N/A',
        'dataset_test': 'Own/team hard dataset - final reporting scope',
        'balanced_accuracy': None,
        'status': 'DONE',
        'notes': 'Per supervisor guidance (Sebastian, meeting July 4/5): own dataset evaluation '
                 'does not need to cover all 6 classes - selecting a subset of well-behaved '
                 'classes and building the narrative around them is acceptable and preferred '
                 'over forcing a single misleading average. DECISION: report own-dataset '
                 'results on 5 classes (cardboard, glass, paper, plastic, trash), excluding '
                 'metal. Metal exclusion is explicitly justified in the report via EXP024 '
                 '(diagnosed composition issue: small objects photographed against large '
                 'natural backgrounds, e.g. tent peg/bottle cap on grass, causing models to '
                 'default to predicting "trash"). This is presented as a documented limitation '
                 'with root-cause analysis, not a hidden gap. Final own-dataset headline '
                 'results (5-class, from EXP025): ResNet-50 zero-shot 43.6% -> fine-tuned '
                 '58.9% (+15.3pts); EfficientNet-B3 34.6% -> 49.3% (+14.7pts); ViT-Small 53.0% '
                 '-> 35.0% (-18.0pts, reported as an architecture-dependent limitation of the '
                 'mitigation strategy, per EXP026).',
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
