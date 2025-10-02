
import os, argparse, json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data import load_dataset, make_generators
from models import MODEL_REGISTRY
from utils.plots import plot_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist","cifar10"])
    parser.add_argument("--model_name", type=str, default="simple_cnn", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    x_train, y_train, x_test, y_test, classes = load_dataset(args.dataset, normalize=True)
    num_classes = len(classes)
    input_shape = x_train.shape[1:]

    # Train/Val split
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    train_gen, val_gen = make_generators(x_tr, y_tr, x_val, y_val, batch_size=args.batch_size, augment=args.augment)

    model = MODEL_REGISTRY[args.model_name](input_shape, num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    ckpt_path = os.path.join(args.models_dir, f"{args.dataset}_{args.model_name}{'_aug' if args.augment else ''}.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs
    )

    # Save best model after training
    model.save(ckpt_path)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    json.dump(report, open(os.path.join(args.outdir, f"{args.dataset}_{args.model_name}_report.json"), "w"), indent=2)

    # Plots
    prefix = f"{args.dataset}_{args.model_name}{'_aug' if args.augment else ''}"
    plot_history(history, outdir=args.outdir, prefix=prefix)

    print(f"Saved model to: {ckpt_path}")
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
