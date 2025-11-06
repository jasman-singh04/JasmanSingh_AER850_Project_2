# Project 2 model_train_A
# Jasman Singh, 501180039
# Due November 5, 2025

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. DATA PROCESSING
DATA_ROOT = "./Project 2 Data/Data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")

INPUT_SHAPE = (500, 500, 3)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3
SEED = 42
OUT_DIR = "./model_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# For reproducibility
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.05,
    zoom_range=0.10,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Validation data is only rescaled
valid_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_gen = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

# Save class mapping for later use in testing
class_indices = train_gen.class_indices
print("Class mapping:", class_indices)
with open(os.path.join(OUT_DIR, "class_indices.json"), "w") as f:
    json.dump(class_indices, f)


# 2. NEURAL NETWORK ARCHITECTURE DESIGN
def build_model_A():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(16, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name="Model_A")


# 3. NETWORK HYPERPARAMETER SELECTIONS
def train_model(model, name):
    model.compile(
        optimizer=optimizers.Adam(1e-4),        # learning rate
        loss="categorical_crossentropy",        # loss function
        metrics=["accuracy"]                     # evaluation metric
    )

    # Callbacks for better training
    ckpt_path = os.path.join(OUT_DIR, f"{name}_best.h5")
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]


    # 4. ACCURACY & LOSS EVALUATION OF THE MODEL
    print(f"\n=== Training {name} ===")
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cbs, verbose=1)

    # Save final model
    model.save(os.path.join(OUT_DIR, f"{name}_final.h5"))
    plot_history(history, name)
    print(f"Training complete for {name}\n")
    return history

# Plot accuracy and loss curves
def plot_history(history, name):
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    loss, val_loss = history.history["loss"], history.history["val_loss"]
    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(10,4))
    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, "b-", label="Train Acc")
    plt.plot(epochs, val_acc, "r-", label="Val Acc")
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, "b-", label="Train Loss")
    plt.plot(epochs, val_loss, "r-", label="Val Loss")
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"performance_{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {name} training curves to {out_path}")


# MAIN
if __name__ == "__main__":
    modelA = build_model_A()
    train_model(modelA, "model_A")