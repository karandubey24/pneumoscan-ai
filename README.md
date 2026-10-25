# PneumoScan AI

AI-Based Image Classification System for Pneumonia Detection from Chest X-rays.
BCA (AI&ML) minor project, Galgotias University, 2026-27.

> Educational prototype. Not a medical device.

## What it does
Classifies a chest X-ray as Normal or Pneumonia using a baseline CNN and a
MobileNetV2 transfer-learning model, and shows Grad-CAM heat maps.

## Dataset
Kaggle "Chest X-Ray Images (Pneumonia)", re-split 70/15/15 by patient to avoid leakage
(`results/split_manifest.csv`).

## Test-set results (878 images, threshold 0.5)
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline CNN | 0.836 | 0.984 | 0.779 | 0.869 | 0.953 |
| MobileNetV2 | 0.958 | 0.978 | 0.961 | 0.970 | 0.990 |

## Run the demo
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    streamlit run app/app.py

## Limitations
Single public dataset, likely paediatric and single-centre; no clinical validation.

## Team
Basu, Karan Dubey, Jatin