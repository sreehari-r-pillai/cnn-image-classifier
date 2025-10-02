
# 🧠 ImageClassificationApp

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-green.svg)
![License](https://img.shields.io/badge/License-MIT-success.svg)

A **complete Deep Learning Image Classification application** built with **TensorFlow/Keras**.  
Supports **Fashion-MNIST** and **CIFAR-10**, with **data augmentation, CNN architectures, training/evaluation scripts, and an optional Streamlit app** for real-time inference.

---

## ✨ Features

- 📂 **Dataset Handling**
  - Auto-download (Fashion-MNIST / CIFAR-10)
  - Preprocessing (resize, normalize, train/val/test split)
  - Data augmentation (rotation, flipping, shifting)

- 🏗️ **Model Development**
  - CNN built **from scratch** with Conv, Pooling, Dropout & BatchNorm
  - Two architectures: `simple_cnn` and `deeper_cnn`
  - Optimized with Adam + EarlyStopping

- 📊 **Evaluation**
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Training vs Validation curves
  - Correct & misclassified samples

- 🧪 **Experimentation**
  - Train with/without augmentation
  - Compare multiple CNNs
  - Save the **best model** for inference

- 🌐 **Deployment (Optional)**
  - **Streamlit app** for inference
  - Upload images → view predictions + probability chart

---

## 📂 Project Structure

```
ImageClassificationApp/
├─ app/                  # Streamlit app
├─ data/                 # Datasets (auto-downloaded)
├─ models/               # Saved models (.h5)
├─ notebooks/            # Jupyter notebooks
├─ results/              # Plots, metrics, confusion matrix
├─ src/                  # Training & evaluation scripts
├─ utils/                # Plotting utilities
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚡ Quick Start

### 1️⃣ Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2️⃣ Train a Model
```bash
# Simple CNN (Fashion-MNIST)
python src/train.py --dataset fashion_mnist --model_name simple_cnn --epochs 10 --batch_size 64

# With augmentation
python src/train.py --dataset fashion_mnist --model_name simple_cnn --epochs 10 --augment

# Deeper CNN (CIFAR-10)
python src/train.py --dataset cifar10 --model_name deeper_cnn --epochs 30 --batch_size 128 --augment
```

### 3️⃣ Evaluate
```bash
python src/evaluate.py --dataset fashion_mnist --model_path models/fashion_mnist_simple_cnn.h5
```

Generates:
- `classification_report.json`
- `confusion_matrix.png`
- `correct_indices.npy` & `misclassified_indices.npy`

### 4️⃣ Run Web App (Optional)
```bash
streamlit run app/streamlit_app.py
```
- Upload an image → See prediction & probability chart

---

## 📊 Expected Results
- **Fashion-MNIST**: ~90–93% accuracy (10–15 epochs, simple CNN)
- **CIFAR-10**: ~70–80% accuracy (30+ epochs, deeper CNN + augmentation)

---

## 🔧 Tech Stack
- 🐍 Python 3.8+
- 🔥 TensorFlow / Keras
- 📚 NumPy, scikit-learn, matplotlib
- 🎨 Streamlit (for web UI)

---

## 📜 License
Licensed under the **MIT License**. Free to use, modify, and share.

---

⭐ If you find this useful, don’t forget to **star this repo**!  

---

## Author : Sreehari R Pillai
