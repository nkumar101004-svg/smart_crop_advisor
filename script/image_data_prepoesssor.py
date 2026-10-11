
from pathlib import Path
import tensorflow as tf

DATA_PATH = Path(r"C:\python project\Dataset")
OUTPUT_PATH = Path(r"C:\python project\data_processed\disease_images")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Image dataset not found: {DATA_PATH}")

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
class_folders = [
    folder for folder in DATA_PATH.iterdir()
    if folder.is_dir()
    and any(
        file.is_file() and file.suffix.lower() in valid_extensions
        for file in folder.rglob("*")
    )
]

if not class_folders:
    raise ValueError(
        "No class folders found. Organize images into disease-name folders."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH / "class_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

normalization = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda images, labels: (normalization(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(
    lambda images, labels: (normalization(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

print("Disease image preprocessing completed.")
print("Classes:", class_names)
print("Image size:", IMAGE_SIZE)
print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Class names saved to:", OUTPUT_PATH / "class_names.txt")
