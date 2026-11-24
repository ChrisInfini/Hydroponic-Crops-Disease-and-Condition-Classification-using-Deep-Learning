# Hydroponic Plant Model Training Scripts

This folder contains the two model-training code files prepared from the code supplied in the conversation.

## Files

- `model_creation.py`: loads the dataset, saves class names, builds the MobileNetV2 model, and compiles it. It does not train or save a trained checkpoint.
- `recovery_training.py`: loads `initial_best.keras`, continues training with callbacks, and saves `final_model.keras`.

## Expected Google Drive layout

```text
Hydroponic_Plant_Disease_AI/
├── unified_dataset/
│   ├── train/
│   ├── validation/
│   └── test/
└── trained_model/
    └── initial_best.keras  # required by recovery_training.py
```

The dataset folders must contain one subfolder per class.

## Environment

These scripts were originally written for Google Colab and use the project path:

`/content/gdrive/MyDrive/Hydroponic_Plant_Disease_AI`

The recovery script imports `google.colab.drive`, so it is intended to run in Colab.

Install TensorFlow in the Colab runtime if it is not already available.

## Important notes

1. The first script only creates and compiles the model. The initial training call and creation of `initial_best.keras` were not present in the supplied first code.
2. The recovery script expects `initial_best.keras` to already exist.
3. The recovery script checks whether a backup directory contains files, but the actual checkpoint recovery behavior is provided by Keras `BackupAndRestore`.
4. The test dataset is loaded in both scripts, but neither supplied script evaluates the model on it.
5. These files preserve the supplied training settings and are not independently validated against the user's Colab environment.
