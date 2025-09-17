# 🧠 Cognitive Load Assessment via 3D-CNN

### *A Computer Vision Technique for NASA-TLX Prediction*

## Badges

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Research](https://img.shields.io/badge/Research-NSUT-red?style=for-the-badge&logo=academia&logoColor=white)](https://nsut.ac.in/)

---

## 🌟 Highlights

- **Non-intrusive cognitive load estimation** from facial video using MediaPipe Face Mesh, 3D-CNN, and explainable AI (SHAP).
- **Real-time prediction** of five NASA-TLX subscales from video streams.
- **Reported performance (overall)**: Test MAE = **3.11** (scale 0–20), R² = **0.43**, across 54 participants and 270 videos.
- **Explainability**: SHAP-based analysis reveals a *Cognitive Load Triad* (eyes, forehead, chin-neck posture) driving predictions.
- **Efficient pipeline**: 468 raw landmarks → 184 regional features → 3D spatiotemporal input to 3D-CNN.

---

## 📊 Performance Metrics

### Training Results
| Metric | Value | Scale |
|--------|-------:|:----:|
| MAE    | **2.06** | 0–20 |
| R²     | **0.71** | — |
| MAPE   | **25.95%** | — |

### Test Results
| Metric | Value | Notes |
|--------|------:|:----|
| MAE    | **3.11** | Scale 0–20 |
| R²     | **0.43** | — |
| Participants | **54** | — |
| Videos | **270** | — |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or newer
- TensorFlow 2.20.0
- MediaPipe 0.10.14
- Recommended: NVIDIA GPU with CUDA/cuDNN compatible with your TF version

Install required packages (example):
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

`requirements.txt` should include at least:
```
tensorflow==2.20.0
mediapipe==0.10.14
numpy
pandas
scikit-learn
shap
opencv-python
```

### Data structure (expected)
```
data/
├── raw/
│   └── {participant_id}/
│       └── video.mp4
├── landmarks184/   # generated landmark-region tensors (preprocessed)
└── labels/
    └── labels.csv  # columns: participant_id, video_id, mental, performance, temporal, frustration, effort
```

### Pipeline overview
```mermaid
graph LR
  A[MP4 Videos] --> B[468 Landmarks (MediaPipe)]
  B --> C[184 Regional Features]
  C --> D[3D-CNN Training]
  D --> E[SHAP Analysis]
```

### Training example
```bash
python train_3dcnn_tlx.py \
  --data_dir data/landmarks184 \
  --labels data/labels/labels.csv \
  --split subject \
  --epochs 100 \
  --batch 16
```

### Generating SHAP explanations
```bash
python shap_explain_tlx.py \
  --checkpoint outputs/checkpoints/best.ckpt \
  --data_dir data/landmarks184 \
  --out_dir outputs/shap
```

---

## 🧬 Model Architecture (summary)

- **Input tensor shape**: `120 × 16 × 12 × 1` (time × height × width × channels) — *adapt to your preprocessing pipeline if different*

**Core 3D-CNN block** (illustrative):

| Layer     | Kernel | Filters | Output Shape (example) |
|-----------|:------:|-------:|:----------------------|
| Conv3D    | 3×3×3  | 32     | 120×16×12×32          |
| Conv3D    | 3×3×3  | 64     | 120×16×12×64          |
| MaxPool3D | 2×1×1  | -      | 60×16×12×64           |
| Conv3D    | 3×3×3  | 128    | 60×16×12×128          |

Follow with global pooling, a few dense layers, and an output head predicting the five NASA-TLX subscales (regression).

**Training details**
- Loss: MSE (or Huber for robustness)
- Optimizer: Adam/AdamW
- Regularization: BatchNorm, Dropout (0.5)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Split: **subject-wise** 80/20 to prevent leakage
- Augmentation: temporal augmentation to standardize FPS and clip lengths

---

## 📍 Facial Regions Analysis (from SHAP)
- **Eyes**: strongly associated with attention/processing — right eye often more sensitive in our dataset.
- **Forehead**: eyebrow slope and muscle tension indicate concentration.
- **Chin–Neck**: postural indicators related to temporal demand and stress.
- **Mouth**: speech-related activity and effort.
- **Cheeks**: asymmetry patterns sometimes correlate with emotional strain.

---

## 📈 NASA-TLX Subscale Breakdown (Train)

| Subscale        | Train MAE | Train R² | Description |
|-----------------|----------:|---------:|-------------|
| Mental Demand   | 1.78      | 0.763    | Cognitive processing requirements |
| Performance     | 1.83      | 0.704    | Task completion effectiveness |
| Temporal Demand | 2.15      | 0.652    | Perceived time pressure |
| Frustration     | 2.18      | 0.618    | Stress and annoyance levels |
| Effort          | 2.62      | 0.520    | Mental exertion required |

---

## 🛠 Technical Specifications

**Hardware (used for reported experiments):**
- GPU: NVIDIA A100-SXM4 (40 GB HBM)
- CPU: Dual AMD EPYC 7743
- RAM: 1 TB
- Storage: High-throughput storage (56 TB)

**Implementation notes**
- Use mixed precision if supported (for speed/memory efficiency).
- Ensure MediaPipe and TensorFlow GPU compatibility for reliable performance.

---

## 📄 License & Ethics
- Data collection: informed consent obtained from all 54 participants.
- Biometric data is sensitive; any sharing must comply with institutional review board (IRB) approvals and applicable data-protection regulations.
- Code and model: licensed under the MIT License (see `LICENSE`).

