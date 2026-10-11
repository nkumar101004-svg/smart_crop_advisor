"""Train and evaluate the plant-disease classifier.

The dataset is split by file (70% train, 15% validation, 15% test) so the
reported test metrics are independent of model selection. Empty class
directories are excluded and reported rather than silently becoming classes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "Dataset"
MODEL_PATH = ROOT / "disease_detection_model.keras"
LEGACY_MODEL_PATH = ROOT / "disease_detection_model.h5"
LABELS_PATH = ROOT / "class_names.txt"
METRICS_PATH = ROOT / "disease_model_metrics.json"
CONFUSION_PATH = ROOT / "disease_confusion_matrix.csv"
IMAGE_SIZE = (160, 160)
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap for a quick local run; 0 uses every valid image.",
    )
    return parser.parse_args()


def seed_everything() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def collect_images(max_per_class: int = 0) -> tuple[list[str], list[int], list[str], list[str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_PATH}")

    folders = sorted(path for path in DATASET_PATH.iterdir() if path.is_dir())
    empty_classes: list[str] = []
    class_names: list[str] = []
    paths: list[str] = []
    labels: list[int] = []

    for folder in folders:
        files = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        if not files:
            empty_classes.append(folder.name)
            continue
        if max_per_class:
            files = files[:max_per_class]
        class_index = len(class_names)
        class_names.append(folder.name)
        paths.extend(str(path) for path in files)
        labels.extend([class_index] * len(files))

    if len(class_names) < 2:
        raise ValueError("At least two non-empty class folders are required.")
    return paths, labels, class_names, empty_classes


def make_splits(paths: list[str], labels: list[int]) -> tuple[np.ndarray, ...]:
    paths_array = np.asarray(paths)
    labels_array = np.asarray(labels, dtype=np.int32)
    train_paths, held_paths, train_labels, held_labels = train_test_split(
        paths_array,
        labels_array,
        test_size=0.30,
        random_state=SEED,
        stratify=labels_array,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        held_paths,
        held_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=held_labels,
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def make_dataset(paths: np.ndarray, labels: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        if training:
            image = tf.image.random_flip_left_right(image, seed=SEED)
            image = tf.image.random_brightness(image, max_delta=0.15, seed=SEED)
        return image, label

    if training:
        path_ds = path_ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return path_ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_v2_preprocessing")(inputs)
    x = augmentation(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def calculate_class_weights(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    return {index: float(total / (num_classes * count)) for index, count in enumerate(counts) if count}


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset, labels: list[str], test_labels: np.ndarray) -> dict:
    probabilities = model.predict(dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(
        test_labels,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(
        confusion_matrix(test_labels, predictions, labels=list(range(len(labels)))),
        index=labels,
        columns=labels,
    ).to_csv(CONFUSION_PATH)
    return {
        "test_accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
            for name, values in report.items()
            if name in labels
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything()
    paths, labels, class_names, empty_classes = collect_images(args.max_per_class)
    splits = make_splits(paths, labels)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = splits
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Excluded empty classes: {empty_classes or 'none'}")
    print(f"Images: train={len(train_paths)}, validation={len(val_paths)}, test={len(test_paths)}")

    train_ds = make_dataset(train_paths, train_labels, args.batch_size, training=True)
    val_ds = make_dataset(val_paths, val_labels, args.batch_size, training=False)
    test_ds = make_dataset(test_paths, test_labels, args.batch_size, training=False)
    model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=1, min_lr=1e-6),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=calculate_class_weights(train_labels, len(class_names)),
        callbacks=callbacks,
        verbose=1,
    )
    metrics = evaluate(model, test_ds, class_names, test_labels)
    metrics["classes"] = class_names
    metrics["empty_classes"] = empty_classes
    metrics["split_counts"] = {"train": len(train_paths), "validation": len(val_paths), "test": len(test_paths)}
    metrics["max_per_class"] = args.max_per_class or None
    metrics["training_history"] = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    print(json.dumps(metrics, indent=2))

    if metrics["test_accuracy"] < 0.50 or metrics["macro_f1"] < 0.45:
        raise RuntimeError(
            "Validation gate failed: model was not saved because test accuracy or macro F1 is too low."
        )
    model.save(MODEL_PATH)
    LABELS_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if LEGACY_MODEL_PATH.exists():
        LEGACY_MODEL_PATH.unlink()
    print(f"Saved validated model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
