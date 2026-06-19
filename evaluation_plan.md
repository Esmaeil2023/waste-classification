# Evaluation Plan

## Goal

The goal of this evaluation is to check how well the waste classification model works and how reliable its predictions are.

## Dataset Split

The dataset will be divided into:

* Training Set: 70%
* Validation Set: 15%
* Test Set: 15%

The test set will only be used for the final evaluation.

## Metrics

### Accuracy

Shows the percentage of correctly classified images.

### Precision

Measures how many predicted images are classified correctly.

### Recall

Measures how many images of a class are found correctly.

### F1-Score

Combines Precision and Recall into one score.

### AUROC

Measures how well the model can separate between different classes.

### ECE (Expected Calibration Error)

Measures if the confidence of the model matches its real accuracy.

## Visualization

### Confusion Matrix

Shows which classes are classified correctly and which classes are confused with each other.

### ROC Curve

Used to visualize the classification performance of the model.

### Reliability Diagram

Used to check if the model confidence is well calibrated.

## OOD Evaluation

The model will also be tested on unseen waste images to see how well it generalizes to new data.

## Tools

The evaluation will be done using:

* Python
* PyTorch
* Scikit-learn
* Matplotlib
