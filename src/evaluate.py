
import os, argparse, json, random
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from utils.plots import save_confusion_matrix
from data import load_dataset, CLASS_NAMES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist","cifar10"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    x_train, y_train, x_test, y_test, classes = load_dataset(args.dataset, normalize=True)

    model = tf.keras.models.load_model(args.model_path)
    probs = model.predict(x_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    with open(os.path.join(args.outdir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    save_confusion_matrix(y_test, y_pred, classes, os.path.join(args.outdir, "confusion_matrix.png"))

    # Save some correctly and misclassified samples
    rng = list(range(len(y_test)))
    random.shuffle(rng)
    correct_idx = [i for i in rng if y_pred[i] == y_test[i]][:args.num_samples]
    wrong_idx = [i for i in rng if y_pred[i] != y_test[i]][:args.num_samples]

    np.save(os.path.join(args.outdir, "correct_indices.npy"), np.array(correct_idx))
    np.save(os.path.join(args.outdir, "misclassified_indices.npy"), np.array(wrong_idx))

    print("Saved confusion matrix and sample indices in results/.")

if __name__ == "__main__":
    main()
