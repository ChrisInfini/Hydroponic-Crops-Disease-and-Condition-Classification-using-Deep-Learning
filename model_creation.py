import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.set_visible_devices([], "GPU")

print("TensorFlow:", tf.__version__)
print("Training device: CPU")

PROJECT_DIR = Path(
    "/content/gdrive/MyDrive/Hydroponic_Plant_Disease_AI"
)

DATA_DIR = PROJECT_DIR / "unified_dataset"
MODEL_DIR = PROJECT_DIR / "trained_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_MODEL = MODEL_DIR / "initial_best.keras"
FINAL_MODEL = MODEL_DIR / "hydroponic_plant_final.keras"

#PARAMTERS
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42


train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("\nNumber of classes:", NUM_CLASSES)
print("First 10 classes:", class_names[:10])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

base_model = keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

#FREEZE
base_model.trainable = False

inputs = keras.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nModel created successfully!")

