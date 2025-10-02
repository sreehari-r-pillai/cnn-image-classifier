
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(history, outdir="results", prefix="run"):
    os.makedirs(outdir, exist_ok=True)
    # Loss
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{prefix}_loss.png"), bbox_inches="tight")
    plt.close()

    # Accuracy
    if "accuracy" in history.history or "acc" in history.history:
        acc_key = "accuracy" if "accuracy" in history.history else "acc"
        val_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
        plt.figure()
        plt.plot(history.history.get(acc_key, []), label="train")
        if val_key in history.history:
            plt.plot(history.history[val_key], label="val")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"{prefix}_accuracy.png"), bbox_inches="tight")
        plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, outpath):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(8,8))
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
