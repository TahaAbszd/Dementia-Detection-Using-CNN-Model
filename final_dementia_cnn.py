"""Final project-ready CNN pipeline for Dementia MRI classification.

Usage examples:
  python final_dementia_cnn.py --data-dir data/train --epochs 30
  python final_dementia_cnn.py --data-dir data/train --predict data/new_cases/sample.jpg --weights artifacts/best_dementia_cnn.keras
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, regularizers


@dataclass
class Config:
    data_dir: Path
    img_size: tuple[int, int] = (128, 128)
    batch_size: int = 16
    epochs: int = 40
    val_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42
    artifacts_dir: Path = Path("artifacts")


class DementiaTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.autotune = tf.data.AUTOTUNE
        tf.keras.utils.set_random_seed(cfg.seed)

        self.artifacts_dir = cfg.artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.artifacts_dir / "best_dementia_cnn.keras"
        self.final_model_path = self.artifacts_dir / "dementia_cnn_final.keras"

        self.class_names: list[str] = []
        self.model: keras.Model | None = None

    def load_datasets(self):
        if not self.cfg.data_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.cfg.data_dir.resolve()}")

        full_ds = tf.keras.utils.image_dataset_from_directory(
            self.cfg.data_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=self.cfg.img_size,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            seed=self.cfg.seed,
        )

        self.class_names = full_ds.class_names
        full_batches = tf.data.experimental.cardinality(full_ds).numpy()

        test_batches = max(1, int(full_batches * self.cfg.test_split))
        val_batches = max(1, int(full_batches * self.cfg.val_split))
        train_batches = full_batches - test_batches - val_batches
        if train_batches < 1:
            raise ValueError("Dataset is too small for current split configuration.")

        train_ds = full_ds.take(train_batches)
        rest_ds = full_ds.skip(train_batches)
        val_ds = rest_ds.take(val_batches)
        test_ds = rest_ds.skip(val_batches)

        print("Classes:", self.class_names)
        print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
        print(f"Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
        print(f"Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")

        augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.08),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ],
            name="data_augmentation",
        )

        normalization = layers.Rescaling(1.0 / 255)

        def prepare(ds, training=False):
            ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=self.autotune)
            if training:
                ds = ds.map(
                    lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=self.autotune,
                )
            return ds.prefetch(self.autotune)

        return prepare(train_ds, training=True), prepare(val_ds), prepare(test_ds)

    def build_model(self):
        l2 = regularizers.L2(1e-4)
        model = keras.Sequential(
            [
                layers.Input(shape=(*self.cfg.img_size, 3)),
                layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2),
                layers.BatchNormalization(),
                layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2),
                layers.MaxPooling2D(),
                layers.Dropout(0.25),
                layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=l2),
                layers.BatchNormalization(),
                layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=l2),
                layers.MaxPooling2D(),
                layers.Dropout(0.25),
                layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=l2),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                layers.Dropout(0.30),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu", kernel_regularizer=l2),
                layers.Dropout(0.40),
                layers.Dense(len(self.class_names), activation="softmax"),
            ],
            name="dementia_cnn",
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return model

    def train(self, train_ds, val_ds):
        if self.model is None:
            raise RuntimeError("Model is not built.")

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, monitor="val_loss", save_best_only=True),
        ]

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def plot_learning_curves(self, history):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history.history["loss"], label="Train Loss")
        ax[0].plot(history.history["val_loss"], label="Val Loss")
        ax[0].set_title("Loss Curve")
        ax[0].set_xlabel("Epoch")
        ax[0].legend()

        ax[1].plot(history.history["accuracy"], label="Train Acc")
        ax[1].plot(history.history["val_accuracy"], label="Val Acc")
        ax[1].set_title("Accuracy Curve")
        ax[1].set_xlabel("Epoch")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(self.artifacts_dir / "learning_curves.png", dpi=150)
        plt.close()

    def evaluate(self, test_ds):
        if self.model is None:
            raise RuntimeError("Model is not built.")

        test_loss, test_acc = self.model.evaluate(test_ds, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        y_true, y_pred = [], []
        for x_batch, y_batch in test_ds:
            probs = self.model.predict(x_batch, verbose=0)
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))
            y_pred.extend(np.argmax(probs, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Test Set)")
        plt.tight_layout()
        plt.savefig(self.artifacts_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=4)
        report_path = self.artifacts_dir / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(report)

    def save(self):
        if self.model is None:
            raise RuntimeError("Model is not built.")
        self.model.save(self.final_model_path)
        print("Model saved to:", self.final_model_path)
        print("Best checkpoint:", self.checkpoint_path)

    def predict_image(self, image_path: Path):
        if self.model is None:
            raise RuntimeError("Model is not loaded/built.")

        img = keras.utils.load_img(image_path, target_size=self.cfg.img_size)
        x = keras.utils.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx]) * 100

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{self.class_names[idx]} ({confidence:.2f}%)")
        out_file = self.artifacts_dir / f"prediction_{image_path.stem}.png"
        plt.savefig(out_file, dpi=150)
        plt.close()

        print(f"Prediction: {self.class_names[idx]} | Confidence: {confidence:.2f}%")
        print(f"Saved preview to: {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate CNN for dementia MRI classification.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"), help="Dataset root directory.")
    parser.add_argument("--img-size", type=int, nargs=2, default=(128, 128), metavar=("W", "H"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--weights", type=Path, default=None, help="Optional .keras weights/model path for inference.")
    parser.add_argument("--predict", type=Path, default=None, help="Single image path for prediction.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
    )

    trainer = DementiaTrainer(cfg)
    train_ds, val_ds, test_ds = trainer.load_datasets()
    trainer.build_model()

    if args.weights and args.weights.exists():
        trainer.model = keras.models.load_model(args.weights)
        print(f"Loaded model from: {args.weights}")

    if args.predict:
        trainer.predict_image(args.predict)
        return

    history = trainer.train(train_ds, val_ds)
    trainer.plot_learning_curves(history)
    trainer.evaluate(test_ds)
    trainer.save()


if __name__ == "__main__":
    main()
