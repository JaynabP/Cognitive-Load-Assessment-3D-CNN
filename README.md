<div align="center">

# 🧠 **Cognitive Load Assessment via 3D-CNN**

### *A Novel Computer Vision Technique for Real-time NASA-TLX Prediction*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Research](https://img.shields.io/badge/Research-NSUT-red?style=for-the-badge&logo=academia&logoColor=white)](https://nsut.ac.in/)

*Pinaki Chakraborty, Jaynab, Harshita Sharma, Ritu Sibal, Rudresh Dwivedi*  
**Netaji Subhas University of Technology, New Delhi**

---

</div>

## 🌟 **Highlights**

> **Non-intrusive cognitive load estimation from facial video using MediaPipe Face Mesh, 3D-CNN, and explainable AI**

- 🎯 **Real-time prediction** of 5 NASA-TLX subscales from video streams
- 🏆 **State-of-the-art performance**: Test MAE **3.11**, R² **0.43** across 54 participants 
- 🔍 **Explainable AI** with SHAP revealing the "**Cognitive Load Triad**"
- ⚡ **Efficient pipeline**: 468 → 184 landmarks → 3D spatiotemporal features
- 📊 **Robust methodology**: Subject-wise 80/20 split, temporal augmentation

---

## 📊 **Performance Metrics**

<div align="center">

### 🎯 **Training Results**
| Metric | Value | Scale |
|--------|-------|-------|
| **MAE** | **2.06** | 0-20 |
| **R²** | **0.71** | - |
| **MAPE** | **25.95%** | - |

### 🔬 **Test Results** 
| Metric | Value | Scale |
|--------|-------|-------|
| **MAE** | **3.11** | 0-20 |
| **R²** | **0.43** | - |
| **Participants** | **54** | - |
| **Videos** | **270** | - |

</div>

---

## 🚀 **Quick Start**

### 📋 **Prerequisites**
### 📋 **Prerequisites**
### 📂 **Data Structure**
data/
├── raw/
│ └── {participant_id}/video.mp4
### 📂 **Data Structure**

graph LR
A[📹 MP4 Videos] --> B[🎯 468 Landmarks]
B --> C[🎨 184 Regions]
C --> D[🧠 3D-CNN Training]
D --> E[📈 SHAP Analysis]

text
style A fill:#e1f5fe
style B fill:#f3e5f5
<div align="center">
#### **Step 3**: Train 3D-CNN Model
python train_3dcnn_tlx.py --data_dir data/landmarks184 --labels data/labels/labels.csv --split subject --epochs 100 --batch 16

text

#### **Step 4**: Generate SHAP Explanations
python shap_explain_tlx.py --checkpoint outputs/checkpoints/best.ckpt --data_dir data/landmarks184 --out_dir outputs/shap

text

---

## 🧬 **Model Architecture**

<div align="center">

### 🎯 **Input Tensor**: `120 × 16 × 12 × 1`
#### **Step 1**: Extract 468 Facial Landmarks
| **Conv3D** | 3×3×3 | 32 | 120×16×12×32 |
| **Conv3D** | 3×3×3 | 64 | 120×16×12×64 |
| **MaxPool3D** | 2×1×1 | - | 60×16×12×64 |
| **Conv3D** | 3×3×3 | 128 | 60×16×12×128 |
#### **Step 2**: Select 184 Regional Features  

</div>

---
#### **Step 3**: Train 3D-CNN Model

<div align="center">

| Class | Region | Indicators | Significance |
#### **Step 4**: Generate SHAP Explanations

</div>

### 📍 **Facial Regions Analysis**
- **Eyes**: Attention and cognitive effort indicators (right eye more sensitive)
- **Forehead**: Concentration markers via eyebrow slope and muscle tension  
- **Chin-Neck**: Strongest postural stress response to temporal demand
- **Mouth**: Speech dynamics reflecting mental task load
- **Cheeks**: Asymmetry patterns indicating emotional strain

---

## 📈 **NASA-TLX Subscale Performance**

<div align="center">

| Subscale | Train MAE | Train R² | Description |
|----------|-----------|----------|-------------|
| **Mental Demand** | **1.78** | **0.763** | 🧠 Cognitive processing requirements |
| **Performance** | **1.83** | **0.704** | ✅ Task completion effectiveness |
| **Temporal Demand** | 2.15 | 0.652 | ⏰ Time pressure perception |
| **Frustration** | 2.18 | 0.618 | 😤 Stress and annoyance levels |
| **Effort** | **2.62** | **0.520** | 💪 Mental exertion required |

</div>

---

## 🛠️ **Technical Specifications**

### 💻 **Hardware Environment**
- **GPU**: NVIDIA A100-SXM4 (40GB HBM)
- **CPU**: Dual AMD EPYC 7743 (256 logical cores)
- **RAM**: 1TB system memory  
- **Storage**: 56TB high-throughput storage

### 🔧 **Key Features**
- **Subject-wise split** prevents data leakage across participants
- **Temporal augmentation** standardizes FPS and clip duration
- **Batch normalization** + **dropout (0.5)** for regularization
- **Early stopping** + **ReduceLROnPlateau** for optimal training
- **SHAP GradientExplainer** for model interpretability

---

## 📚 **Citation**

@article{chakraborty2025cognitive,
title={A Novel Computer Vision based Technique for Assessing Cognitive Load},
author={Chakraborty, Pinaki and Jaynab and Sharma, Harshita and Sibal, Ritu and Dwivedi, Rudresh},
journal={Department of Computer Science and Engineering},
institution={Netaji Subhas University of Technology},
year={2025},
address={New Delhi, India}
}

text

---

## 📄 **License & Ethics**

- 📋 **Informed consent** obtained from all 54 participants
- 🔒 **Biometric data** requires appropriate approvals for sharing
- 📜 Licensed under MIT License (see `LICENSE`)

---

<div align="center">

### 🌟 **Built with ❤️ at NSUT**

*Advancing Human-Computer Interaction through Explainable AI*

[![Email](https://img.shields.io/badge/Contact-pinaki.chakraborty%40nsut.ac.in-blue?style=for-the-badge&logo=gmail&logoColor=white)](mailto:pinaki.chakraborty@nsut.ac.in)

</div>