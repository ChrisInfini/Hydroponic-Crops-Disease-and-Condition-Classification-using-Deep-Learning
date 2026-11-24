import tensorflow as tf
from tensorflow import keras
from google.colab import drive
from pathlib import Path

drive.mount("/content/gdrive")

PROJECT_DIR = Path(
    "/content/gdrive/MyDrive/Hydroponic_Plant_Disease_AI"
)

DATASET_DIR = PROJECT_DIR / "unified_dataset"
MODEL_DIR = PROJECT_DIR / "trained_model"

BEST_MODEL = MODEL_DIR / "initial_best.keras"
BACKUP_DIR = MODEL_DIR / "training_backup"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

#SAVED
if not BEST_MODEL.exists():
    raise FileNotFoundError(
        f"Best model not found: {BEST_MODEL}"
    )

print("Best model found:", BEST_MODEL)

backup_files = (
    list(BACKUP_DIR.rglob("*"))
    if BACKUP_DIR.exists()
    else []
)

has_backup = any(path.is_file() for path in backup_files)

if has_backup:
    print("Recovery backup found. Attempting to resume from it.")
else:
    print("No recovery backup found.")
    print("Will continue from the best saved model instead.")

#DATASET
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Classes:", len(class_names))
print("Datasets loaded.")

model = keras.models.load_model(BEST_MODEL)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        initial_value_threshold=0.90269,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    keras.callbacks.BackupAndRestore(
        backup_dir=str(BACKUP_DIR)
    )
]

print("\nStarting recovery training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

FINAL_MODEL = MODEL_DIR / "final_model.keras"

model.save(FINAL_MODEL)

print("\nTraining run finished.")
print("Final model:", FINAL_MODEL)
print("Best checkpoint:", BEST_MODEL)
