# AI Smart Crop Advisor and Plant Disease Detection System

A professional Streamlit frontend for the crop model and image dataset in this project.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The repository already contains:

- `random_forest_crop_model.pkl` — the saved crop recommendation model
- `model_columns.pkl` — the exact one-hot encoded feature schema used by the model
- `crop_recommendation_dataset_realistic.csv` — the crop model reference dataset
- `Dataset/` — plant disease image folders and their class names

The crop page uses the model's saved feature columns and reproduces the preprocessing from
`script/train.py`: missing categorical/numeric values are handled during training and categorical
features are one-hot encoded with `drop_first=True`.

## Optional disease model

The disease page is fully wired but does not fabricate results. Train the validated transfer
learning model with:

```powershell
.\.venv\Scripts\python.exe script\imagetrain.py
```

The script checks image files, excludes empty class folders, creates deterministic stratified
train/validation/test splits, trains MobileNetV2 with ImageNet weights, writes
`disease_model_metrics.json` and `disease_confusion_matrix.csv`, and saves the model only when
the test accuracy is at least 50% and macro F1 is at least 45%. Empty class folders are excluded, so
the model's class count always matches the non-empty dataset.
The model includes its own 160 x 160 RGB MobileNetV2 preprocessing layer; the app uses that
model preprocessing and rejects mismatched label counts.

Prediction history is stored locally in `prediction_history.csv`; uploaded images are not saved.
All metrics and class lists shown by the app come from local files or dataset metadata. The app
does not calculate disease accuracy from user uploads.
