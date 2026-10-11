import pandas as pd
import numpy as np

# 1. Load the dataset
file_name = 'seasonal_agriculture_performance_dataset (3).csv'
df = pd.read_csv(file_name)
print("Original Dataset Shape:", df.shape)

# 2. Drop Unnecessary Columns
# Farm_ID is a unique identifier and does not contribute to analysis
if 'Farm_ID' in df.columns:
    df = df.drop('Farm_ID', axis=1)
    print("Dropped 'Farm_ID' column.")

# 3. Handle Duplicates
duplicates_count = df.duplicated().sum()
if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates_count} duplicate rows.")
else:
    print("No duplicate rows found.")

# 4. Handle Missing Values
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Strategy: Fill missing numerical values with the median of their respective columns
# Median is preferred over mean as it is robust to outliers
cols_with_missing = ['Rainfall_mm', 'Soil_Moisture_pct', 'Yield_Tonnes_Ha']

for col in cols_with_missing:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "total missing values remaining.")

# 5. Handle Potential Outliers (Optional but recommended)
# Example: Capping extreme outliers in 'Total_Cost_INR' using the IQR method
Q1 = df['Total_Cost_INR'].quantile(0.25)
Q3 = df['Total_Cost_INR'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap values outside the bounds
df['Total_Cost_INR'] = np.where(
    df['Total_Cost_INR'] > upper_bound, upper_bound, df['Total_Cost_INR'])
df['Total_Cost_INR'] = np.where(
    df['Total_Cost_INR'] < lower_bound, lower_bound, df['Total_Cost_INR'])

# 6. Standardize Text Formatting (Categorical Columns)
# Ensure all text is consistently formatted (e.g., stripped of leading/trailing spaces and capitalized)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].str.strip().str.title()

# 7. Save the Cleaned Dataset
cleaned_file_name = 'cleaned_agriculture_dataset.csv'
df.to_csv(cleaned_file_name, index=False)
print(f"\nCleaned dataset saved as '{cleaned_file_name}'.")
print("Cleaned Dataset Shape:", df.shape)
