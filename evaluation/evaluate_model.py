import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from google.colab import drive
drive.mount("/content/gdrive")

PROJECT_DIR = Path(
    "/content/gdrive/MyDrive/Hydroponic_Plant_Disease_AI"
)

DATASET_DIR = PROJECT_DIR / "unified_dataset"
MODEL_DIR = PROJECT_DIR / "trained_model"
BEST_MODEL = MODEL_DIR / "initial_best.keras"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 16

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not BEST_MODEL.exists():
    raise FileNotFoundError(f"Model file not found: {BEST_MODEL}")

print("\nLoading test dataset...")

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

print("\nNumber of test classes:", num_classes)
print("Class-index mapping:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("\nLoading trained model...")
model = keras.models.load_model(BEST_MODEL)
print("Model loaded successfully.")
model.summary()

if model.output_shape[-1] != num_classes:
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes, but "
        f"the test dataset contains {num_classes}. Check class mapping."
    )

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print("\nGenerating predictions...")

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
predictions = model.predict(test_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

if len(y_true) != len(y_pred):
    raise RuntimeError("Mismatch between actual labels and predictions.")

overall_accuracy = accuracy_score(y_true, y_pred)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)
print(f"Test Loss          : {test_loss:.4f}")
print(f"Test Accuracy      : {overall_accuracy * 100:.2f}%")
print(f"Macro Precision    : {macro_precision:.4f}")
print(f"Macro Recall       : {macro_recall:.4f}")
print(f"Macro F1-score     : {macro_f1:.4f}")
print(f"Weighted Precision : {weighted_precision:.4f}")
print(f"Weighted Recall    : {weighted_recall:.4f}")
print(f"Weighted F1-score  : {weighted_f1:.4f}")
print(f"Test images        : {len(y_true)}")
print(f"Number of classes  : {num_classes}")

print("\n" + "=" * 55)
print("CLASSIFICATION REPORT")
print("=" * 55)

report_text = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    digits=4,
    zero_division=0
)
print(report_text)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report_text)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

with open(OUTPUT_DIR / "classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=4)

cm = confusion_matrix(
    y_true, y_pred, labels=np.arange(num_classes)
)

fig, ax = plt.subplots(
    figsize=(max(14, num_classes * 0.55), max(11, num_classes * 0.45))
)

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_title("Confusion Matrix - Test Dataset")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)

if num_classes <= 25:
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7
            )

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)
row_totals = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0
)

fig, ax = plt.subplots(
    figsize=(max(14, num_classes * 0.55), max(11, num_classes * 0.45))
)

im = ax.imshow(
    cm_normalized,
    interpolation="nearest",
    cmap="Blues",
    vmin=0,
    vmax=1
)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_title("Normalized Confusion Matrix - Test Dataset")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "normalized_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)
class_precision, class_recall, class_f1, class_support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        zero_division=0
    )
)

class_results = {}

print("\n" + "=" * 55)
print("PER-CLASS PERFORMANCE")
print("=" * 55)

for i, class_name in enumerate(class_names):
    class_results[class_name] = {
        "precision": float(class_precision[i]),
        "recall": float(class_recall[i]),
        "f1_score": float(class_f1[i]),
        "support": int(class_support[i])
    }

    print(
        f"{class_name:40s} | "
        f"Precision: {class_precision[i]:.4f} | "
        f"Recall: {class_recall[i]:.4f} | "
        f"F1: {class_f1[i]:.4f} | "
        f"Images: {class_support[i]}"
    )

with open(OUTPUT_DIR / "class_accuracy.json", "w") as f:
    json.dump(class_results, f, indent=4)
plant_results = defaultdict(list)

for class_name, metrics in class_results.items():
    # Assumes directory names use the format Plant_Condition.
    plant_name = class_name.split("_", 1)[0]
    plant_results[plant_name].append(metrics["recall"])

plant_wise_results = {}

print("\n" + "=" * 55)
print("PLANT-WISE MEAN CLASS RECALL")
print("=" * 55)

for plant, values in sorted(plant_results.items()):
    mean_recall = float(np.mean(values))
    plant_wise_results[plant] = mean_recall
    print(f"{plant:25s}: {mean_recall * 100:.2f}%")

summary = {
    "model": BEST_MODEL.name,
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "test_accuracy": float(overall_accuracy),
    "test_loss": float(test_loss),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1_score": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1_score": float(weighted_f1),
    "number_of_test_images": int(len(y_true)),
    "number_of_classes": int(num_classes),
    "class_names": class_names,
    "plant_wise_mean_class_recall": plant_wise_results
}

with open(OUTPUT_DIR / "test_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print("\n" + "=" * 55)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 55)
print("\nResults saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print("-", path.name)
