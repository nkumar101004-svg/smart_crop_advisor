
import pandas as pd
from pathlib import Path

# Aapki original CSV file ka path
csv_file = Path(
    r"C:\python project\Dataset\seasonal_agriculture_performance_dataset (3).csv"
)

# Check karo ki file available hai ya nahi
if not csv_file.exists():
    print("Dataset file nahi mili. Path check karo.")
    raise SystemExit(1)

# Dataset load karo
df = pd.read_csv(csv_file)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCrop Types:")
print(df["Crop"].value_counts())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nModule 1 completed successfully!")
