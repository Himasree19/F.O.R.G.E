import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load Merged CSV
merged_csv_path = r"C:\Users\LENOVO\Desktop\F.O.R.G.E-wbl\F.O.R.G.E-main\backend\continuous_learning\image\merged_csv\merged_dataset.csv"
df = pd.read_csv(merged_csv_path, low_memory=False)

# Clean Data
df = df.replace([np.inf, -np.inf], np.nan)

# Clean only rows where 'label' is missing
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

y = df["label"]

# Drop non-feature metadata columns
metadata_cols = [
    "image_path", "label", "image_id", "verified_label",
    "predicted_label", "user_feedback", "timestamp",
    "status", "batch_id", "model_version", "prediction_confidence"
]
cols_to_drop = [c for c in metadata_cols if c in df.columns]

X = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])

# Clean remaining NaN values in feature columns with median
X = X.fillna(X.median())

# Give 50x higher weight to the replay samples (the last rows added)
weights = np.ones(len(df))
# Boost weight of replay rows so the model doesn't ignore them
weights[-79:] = 50.0  

# Train Model
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,   # Reduced from 4 to better learn small replay subsets
    min_samples_leaf=1,    # Reduced from 4 so replay trees fit tightly
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    oob_score=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining Random Forest Model v2 with weighted samples...")
rf_model.fit(X, y, sample_weight=weights)

# Save Single v2 Model
models_dir = r"C:\Users\LENOVO\Desktop\F.O.R.G.E-wbl\F.O.R.G.E-main\backend\continuous_learning\image\models"
os.makedirs(models_dir, exist_ok=True)
model_v2_path = os.path.join(models_dir, "Image_rf_fusion_v2.pkl")

joblib.dump(rf_model, model_v2_path)
print(f"\nModel v2 successfully trained and saved to: {model_v2_path}")