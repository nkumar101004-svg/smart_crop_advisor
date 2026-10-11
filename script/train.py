import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Dataset load karein (Exact path ke sath)
df = pd.read_csv(
    r'C:\python project\crop_recommendation_dataset_realistic.csv')

# 2. Agar Target ('Crop') mein hi koi missing value hai, toh us row ko hata dein
df = df.dropna(subset=['Crop'])

# 3. Features (X) aur Target (y) alag karein
X = df.drop(columns=['Crop'])
y = df['Crop']

# 4. Missing Values (Khali Data) ko Fix Karein
num_cols = X.select_dtypes(include=['number']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

cat_cols = X.select_dtypes(include=['object']).columns
X[cat_cols] = X[cat_cols].fillna('Unknown')

# 5. Text ko Numbers mein convert karein (One-Hot Encoding)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 6. Data ko Train (80%) aur Test (20%) mein split karein
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("-" * 50)
print(f"Total Data: {len(df)} rows")
print(f"Features (Columns) after encoding text: {X_train.shape[1]}")
print(f"Training Data: {X_train.shape[0]} rows")
print(f"Testing Data: {X_test.shape[0]} rows")
print("-" * 50)

# 7. Random Forest Model banayein aur Train karein
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nRandom Forest Model training complete!")

# 8. Test Data par prediction karein aur accuracy check karein
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Detailed Report
print("Detailed Performance Report:")
print(classification_report(y_test, y_pred))

# 9. Model aur Columns list ko save karein
joblib.dump(model, 'random_forest_crop_model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')

print("\nSaved model as 'random_forest_crop_model.pkl'")
print("Saved columns list as 'model_columns.pkl'")
