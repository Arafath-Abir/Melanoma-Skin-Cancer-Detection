Melanoma Skin Cancer Detection
===============================

Overview
--------
This repository contains a complete, notebook-driven pipeline for binary classification of dermoscopic skin lesions (benign vs malignant/melanoma). The work is implemented in a single Jupyter/Colab notebook and uses transfer learning with modern CNN backbones, basic class-imbalance handling, and model explainability.

What’s Inside
-------------
- Notebook-first workflow (tested in Google Colab with Google Drive mounted).
- Directory-based image loading via Keras ImageDataGenerator.
- Transfer Learning backbones referenced in the notebook: MobileNetV2 and InceptionV3 (with a custom classification head).
- Evaluation utilities for confusion matrix and classification report.
- Explainability with LIME (sample, extendable to Grad-CAM).
- Multiple runs recorded in the notebook; observed accuracies around 0.84–0.96 depending on split and backbone.

Repository Layout (suggested)
-----------------------------
- notebooks/
  - model_with_outputs_github.ipynb   # cleaned, GitHub-renderable (use this on GitHub)
  - model.ipynb                       # original notebook (may contain widget metadata)
- melanoma_cancer_dataset/            # your dataset root (not tracked in git)
  - train/
    - benign/
    - malignant/
  - val/                              # optional if using validation_split
    - benign/
    - malignant/
- assets/                             # optional: export plots here (confusion matrix, ROC, etc.)
- checkpoints/                        # optional: saved models (add to .gitignore)
- requirements.txt
- README.md

Dataset Expectations
--------------------
- Input: RGB dermoscopic images.
- Target size: 128x128 (resized in the loader).
- Normalization: rescale to [0, 1] via ImageDataGenerator (rescale=1/255).
- Split: typical 90/10 using validation_split=0.1, or an explicit val/ directory.
- Loading: flow_from_directory with batch_size=64, shuffle=True, color_mode='rgb'.
- Classes: two folders named "benign" and "malignant".

Environment and Requirements
----------------------------
Python 3.x with the following libraries (adjust versions as needed):
- tensorflow / keras
- numpy, pandas
- matplotlib, seaborn
- scikit-learn
- opencv-python (cv2) and/or scikit-image
- imbalanced-learn (optional, for class imbalance strategies)
- lime (for explainability)

Example requirements.txt
------------------------
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
scikit-image
imbalanced-learn
lime

Quick Start
-----------
1) Create a virtual environment and install dependencies:
   python -m venv .venv
   (Windows) .venv\Scripts\activate
   (macOS/Linux) source .venv/bin/activate
   pip install -r requirements.txt

2) Place your dataset under melanoma_cancer_dataset/ following the structure above.

3) Open the notebook:
   - notebooks/model_with_outputs_github.ipynb  (recommended for GitHub viewing)
   - or notebooks/model.ipynb

4) In the notebook:
   - Update the dataset path if needed.
   - Run all cells to:
     • prepare generators (train/validation)
     • build the model (MobileNetV2 or InceptionV3 base + custom head)
     • train with early stopping/checkpointing (if enabled)
     • evaluate and print metrics
     • generate plots (learning curves, confusion matrix, ROC) if desired
     • run LIME explanations on a few samples (optional)

Training Configuration (as used in the notebook)
------------------------------------------------
- Image size: 128x128, RGB
- Batch size: 64
- Validation split: 0.1 (via ImageDataGenerator)
- Optimizer/Loss: Keras standard setup for binary classification
- Class imbalance: augmentation and/or class weights (imbalanced-learn available)
- Backbones: MobileNetV2 and InceptionV3 (transfer learning, custom top layers)

Evaluation
----------
- Metrics: accuracy, precision, recall, F1-score (per class via classification_report)
- Confusion matrix: printed/drawn in the notebook
- ROC-AUC: optional, can be added alongside probability outputs
- Observed accuracy (notebook runs): ~0.84, ~0.93, up to ~0.96 depending on configuration and split

Inference Example (update paths as needed)
------------------------------------------
import numpy as np
import tensorflow as tf
import cv2

IMG_SIZE = (128, 128)

def preprocess_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE) / 255.0
    return np.expand_dims(img, axis=0)

model = tf.keras.models.load_model("checkpoints/best_model.h5")  # adjust path if different
x = preprocess_img("samples/lesion_001.jpg")
pred = model.predict(x)[0][0]  # sigmoid output for binary classification
print("Malignant probability:", float(pred))

Explainability
--------------
- LIME: used in the notebook to provide local explanations for individual predictions.
- Grad-CAM (optional): recommended to visualize salient regions learned by the CNN and confirm focus on lesion areas.

Reproducibility Notes
---------------------
- Keep hyperparameters (image size, batch, epochs, learning rate, backbone) grouped near the top of the notebook.
- Pin versions in requirements.txt.
- Save the best model and training logs to checkpoints/.
- For sharing static results, export the notebook to HTML using nbconvert if needed.

Acknowledgements
----------------
- TensorFlow/Keras community and documentation.
- Public melanoma datasets and their contributors (add your dataset source and citation here).
