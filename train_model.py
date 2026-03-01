import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# Leaf Venation CNN Classifier - Training on Venation Maps
# Trains a CNN on venation maps with heavy augmentation for robustness.
# Requires: pip install tensorflow
# =====================================================================

# --- Configuration ---
VENATION_DIR = "./venation_maps"
MODEL_PATH = "leaf_cnn_model.keras"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)

# --- Step 1: Load venation maps ---
print("=" * 60)
print("Step 1: Loading venation maps...")
print("=" * 60)

train_dataset = keras.utils.image_dataset_from_directory(
    VENATION_DIR,
    validation_split=0.2,
    subset="training",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

val_dataset = keras.utils.image_dataset_from_directory(
    VENATION_DIR,
    validation_split=0.2,
    subset="validation",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

class_names = train_dataset.class_names
print(f"  Classes: {class_names}")
print(f"  Training batches: {len(train_dataset)}")
print(f"  Validation batches: {len(val_dataset)}")

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# --- Step 2: Build CNN model with heavy augmentation ---
print(f"\n{'=' * 60}")
print("Step 2: Building CNN model with heavy augmentation...")
print(f"{'=' * 60}")

# Heavy augmentation to simulate noisy real-world venation maps
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.4),
    layers.RandomZoom((-0.2, 0.3)),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.3),
])

model = keras.Sequential([
    layers.Input(shape=(128, 128, 1)),

    # Augmentation (only during training)
    data_augmentation,

    # Normalize
    layers.Rescaling(1.0 / 255),

    # Add Gaussian noise for robustness to imperfect vein extraction
    layers.GaussianNoise(0.1),

    # Conv Block 1
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Conv Block 2
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Conv Block 3
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Conv Block 4
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.GlobalAveragePooling2D(),

    # Dense layers
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Step 3: Train ---
print(f"\n{'=' * 60}")
print("Step 3: Training CNN (up to 50 epochs with early stopping)...")
print(f"{'=' * 60}")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- Step 4: Evaluate ---
print(f"\n{'=' * 60}")
print("Step 4: Evaluating on validation set...")
print(f"{'=' * 60}")

val_labels = []
val_preds = []

for images, labels in val_dataset:
    preds = model.predict(images, verbose=0)
    val_preds.extend(np.argmax(preds, axis=1))
    val_labels.extend(np.argmax(labels.numpy(), axis=1))

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

accuracy = np.mean(val_labels == val_preds)
print(f"\n  Overall Accuracy: {accuracy * 100:.2f}%\n")

print("  Classification Report:")
print("  " + "-" * 55)
label_names = [c.lower() for c in class_names]
print(classification_report(val_labels, val_preds, target_names=label_names))

print("  Confusion Matrix:")
print("  " + "-" * 55)
cm = confusion_matrix(val_labels, val_preds)
header = "          " + "  ".join(f"{c:>10}" for c in label_names)
print(header)
for i, cn in enumerate(label_names):
    row = f"  {cn:>8} " + "  ".join(f"{cm[i][j]:>10}" for j in range(len(label_names)))
    print(row)

# --- Step 5: Save ---
print(f"\n{'=' * 60}")
print("Step 5: Saving model...")
print(f"{'=' * 60}")

model.save(MODEL_PATH)

import json
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print(f"  CNN model saved to '{MODEL_PATH}'")
print(f"  Class names saved to 'class_names.json'")
print(f"\nDone! Run classifier.py to classify new leaves.")
