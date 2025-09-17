# 🧠 Cognitive Load Assessment via 3D-CNN

### *Real-time NASA-TLX Prediction from Facial Video*

---

This project introduces a **non-intrusive method** to estimate cognitive load from facial videos. Using **MediaPipe Face Mesh** for landmark extraction and a **3D-CNN** for spatiotemporal modeling, the system predicts the five NASA-TLX subscales in real time. 

- 🎯 Real-time inference from video streams
- 🔍 Explainability with **SHAP** to reveal important facial regions
- ⚡ End-to-end efficient pipeline: landmarks → features → 3D-CNN → interpretable outputs

---

## 📊 Performance

| Metric | Train | Test |
|--------|------:|----:|
| MAE    | 2.06  | 3.11 |
| R²     | 0.71  | 0.43 |

---

## 📂 Data Structure

```
data/
├── raw/
│   └── {participant_id}/video.mp4
├── landmarks184/
└── labels/labels.csv
```

---

## 🧬 Model Architecture

- **Input**: Preprocessed landmark tensors (120 × 16 × 12 × 1)
- **Layers**: Stacked Conv3D blocks + MaxPooling + Dense regression head
- **Output**: Predicted scores for the 5 NASA-TLX subscales

---

## 📍 Key Insights

- **Eyes** → attention & focus
- **Forehead** → concentration via muscle tension
- **Chin/Neck** → posture & stress indicators
- **Mouth** → speech-related effort
- **Cheeks** → subtle emotional strain

---

## ⚙️ Quick Start

### Installation
```bash
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

### Training
```bash
python train_3dcnn_tlx.py \
  --data_dir data/landmarks184 \
  --labels data/labels/labels.csv \
  --split subject \
  --epochs 100 \
  --batch 16
```

### Explainability
```bash
python shap_explain_tlx.py \
  --checkpoint outputs/checkpoints/best.ckpt \
  --data_dir data/landmarks184 \
  --out_dir outputs/shap
```

---

## 📄 License & Ethics

- Informed consent obtained from all participants
- Biometric data must be handled responsibly
- Code is open-sourced under **MIT License**

---

