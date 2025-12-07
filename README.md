# Deepfake-Detection-Hybrid
# Dual-Domain Explainable Deepfake Detection 🕵️‍♂️🚫

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Abstract
This project presents a unified, dual-domain framework for detecting deepfake images with high accuracy and interpretability. Unlike traditional "black box" models, our approach analyzes both the **Spatial Domain** (visual semantics) and the **Frequency Domain** (spectral artifacts).

By fusing fine-tuned CNN features with statistical frequency descriptors and classifying them using a **Nyström-Approximated SVM**, we achieve state-of-the-art performance while providing transparent forensic evidence via **Grad-CAM** and **SHAP**.

## 🚀 Key Features
- **Hybrid Architecture:** Decouples feature extraction (CNN + DFT) from classification (SVM).
- **Dual-Domain Analysis:**
  - **Spatial:** Fine-tuned Xception/EfficientNet backbone captures visual anomalies.
  - **Frequency:** Custom DFT pipeline extracts Mean, Variance, Skewness, and Kurtosis to detect GAN upsampling artifacts.
- **High Accuracy:** Achieved **97.5% Accuracy** on the 140k Real and Fake Faces dataset.
- **Explainability:**
  - **Grad-CAM:** Visualizes *where* the manipulation is (e.g., eyes, mouth).
  - **SHAP:** Visualizes *which* frequency bands contributed to the "Fake" verdict.

## 📊 Methodology

![Architecture Pipeline](images/pipeline_diagram.png)
*Figure 1: The proposed Dual-Domain Hybrid Architecture.*

1.  **Preprocessing:** Images are resized to 224x224. Spatial inputs are normalized [0,1]; Frequency inputs preserve raw [0,255] intensity.
2.  **Feature Extraction:**
    * **Spatial:** Global Average Pooling from a fine-tuned Xception network (2048-dim vector).
    * **Frequency:** Log-Magnitude Spectrum via FFT -> Band-wise Statistical moments (12-dim vector).
3.  **Fusion & Classification:** Features are concatenated (2060-dim), standardized, and classified using a **Nyström-Approximated SVM** (SGD Solver) to handle large-scale data efficiently.

## 🧪 Experimental Results

We evaluated the model on the **140k Real and Fake Faces** dataset (70k FFHQ Real / 70k StyleGAN Fake) and tested on **FaceForensics++**.

| Metric | Real | Fake | Overall |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.97 | 0.98 | **0.98** |
| **Recall** | 0.98 | 0.97 | **0.97** |
| **F1-Score** | 0.97 | 0.98 | **0.98** |
| **Accuracy** | - | - | **97.5%** |

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Explainability
* **Spatial Attention:** The model focuses on high-frequency regions like hair boundaries and eyes.
* **Frequency Importance:** SHAP analysis reveals that **High-Frequency Kurtosis** is a dominant predictor for deepfakes.

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Deepfake-Detection-Hybrid.git](https://github.com/YOUR_USERNAME/Deepfake-Detection-Hybrid.git)
   cd Deepfake-Detection-Hybrid
