import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load the cleaned dataset
df = pd.read_csv('cleaned_agriculture_dataset.csv')

# 2. Select Relevant Features for Crop Recommendation
# We only keep features that are available before planting (soil, weather, geography).
# We drop post-harvest/financial metrics (Yield, Revenue, Cost, Profit, etc.)
features_to_keep = [
    'State', 'District', 'Season', 'Rainfall_mm', 'Avg_Temperature_C',
    'Humidity_pct', 'Sunlight_Hours_Day', 'Soil_pH', 'Soil_Moisture_pct',
    'Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha'
]

X = df[features_to_keep]
y = df['Crop']  # The target variable we want to recommend

# 3. Encode Categorical Target (Crop)
# Convert crop names (Rice, Wheat, etc.) into numbers (0, 1, 2...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Print the crop classes to know what number maps to what crop
crop_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Crop Class Mapping:", crop_mapping)

# 4. Encode Categorical Features in X (State, District, Season)
# Using One-Hot Encoding via pandas get_dummies
X_encoded = pd.get_dummies(
    X, columns=['State', 'District', 'Season'], drop_first=True)

# 5. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 6. Feature Scaling (Standardization)
# Scale numerical features so that distance-based or gradient-based models perform well
scaler = StandardScaler()

# Fit only on training data, then transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easier viewing (optional)
X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"\nPreprocessing for Crop Recommendation Complete!")
print(f"Training features shape: {X_train_final.shape}")
print(f"Testing features shape: {X_test_final.shape}")
