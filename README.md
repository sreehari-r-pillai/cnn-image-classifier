
# Image Classification App (Keras/TensorFlow)

A complete image classification project using **TensorFlow/Keras** on **Fashion-MNIST** or **CIFAR-10** with:
- Data loading & preprocessing [+ augmentation]
- Two CNN architectures
- Training & evaluation scripts
- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix and training curves
- Sample correctly classified and misclassified indices
- Saved models for inference
- Optional **Streamlit** app for image upload & prediction

## Project Structure

```
ImageClassificationApp/
├─ app/
│  └─ streamlit_app.py         # Simple web UI for inference
├─ data/
│  └─ README.md                # Dataset guidance (auto-downloads when training)
├─ models/                     # Saved models (.h5)
├─ notebooks/
│  └─ Training_Experiments.ipynb
├─ results/                    # Plots, metrics, CM
├─ src/
│  ├─ data.py                  # Load/augment datasets
│  ├─ models.py                # Two CNN architectures
│  ├─ train.py                 # Training script (with/without augmentation)
│  └─ evaluate.py              # Evaluation script (report + confusion matrix)
├─ utils/
│  └─ plots.py                 # Plot helpers
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Datasets

- **Fashion-MNIST** (default)
- **CIFAR-10**

Datasets auto-download via Keras the first time you run training/evaluation.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Training

Train two CNNs **with** and **without** augmentation, saving the best model by validation accuracy.

```bash
# Simple CNN on Fashion-MNIST without augmentation
python src/train.py --dataset fashion_mnist --model_name simple_cnn --epochs 10 --batch_size 64

# Simple CNN with augmentation
python src/train.py --dataset fashion_mnist --model_name simple_cnn --epochs 10 --batch_size 64 --augment

# Deeper CNN on CIFAR-10
python src/train.py --dataset cifar10 --model_name deeper_cnn --epochs 30 --batch_size 128 --augment
```

Artifacts:
- Saved model: `models/<dataset>_<model>[_aug].h5`
- Plots: `results/<dataset>_<model>[_aug]_loss.png`, `..._accuracy.png`
- Report: `results/<dataset>_<model>_report.json`

## Evaluation

```bash
python src/evaluate.py --dataset fashion_mnist --model_path models/fashion_mnist_simple_cnn.h5
```

Outputs:
- `results/classification_report.json` (includes precision/recall/F1 per class)
- `results/confusion_matrix.png`
- `results/correct_indices.npy`, `results/misclassified_indices.npy` (indices into test set)
  - You can visualize these in the notebook to show **correctly classified** and **misclassified** samples.

## Notebook

Open `notebooks/Training_Experiments.ipynb` for an end-to-end interactive workflow with plots and quick experiments.

## Streamlit App (Optional)

```bash
streamlit run app/streamlit_app.py
```
- Select dataset mapping and provide a path to a trained `.h5` model in `models/`.
- Upload an image to get the predicted class and probability bar chart.

## Results (Example Targets)

- Fashion-MNIST: ~0.90–0.93 test accuracy (simple CNN, 10–15 epochs).
- CIFAR-10: ~0.70–0.80 test accuracy (deeper CNN, 30+ epochs with augmentation).
  - Results vary by compute, epochs, and augmentation settings.

## Zip-Ready

This repo excludes heavy data and cache files via `.gitignore`. You can zip the folder directly.
Generated on 2025-10-02.
