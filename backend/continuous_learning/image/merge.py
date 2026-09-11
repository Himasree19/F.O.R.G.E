import os
import pandas as pd

# File Paths
features_csv_path = r"C:\Users\LENOVO\Desktop\final forgery\image_forgery\final_features_for_rf.csv"
replay_csv_path = r"C:\Users\LENOVO\Desktop\F.O.R.G.E-wbl\F.O.R.G.E-main\backend\continuous_learning\image\replay\replay_dataset.csv"

merged_dir = r"C:\Users\LENOVO\Desktop\F.O.R.G.E-wbl\F.O.R.G.E-main\backend\continuous_learning\image\merged_csv"
os.makedirs(merged_dir, exist_ok=True)
merged_csv_path = os.path.join(merged_dir, "merged_dataset.csv")

# Load Datasets
df_base = pd.read_csv(features_csv_path)
df_replay = pd.read_csv(replay_csv_path)

# Label Mapping
label_map = {"Fake": 1, "fake": 1, "1": 1, 1: 1, "Real": 0, "real": 0, "0": 0, 0: 0}

if "verified_label" in df_replay.columns:
    df_replay["label"] = df_replay["verified_label"].map(label_map)
elif "label" in df_replay.columns:
    df_replay["label"] = df_replay["label"].map(label_map)

# Fill any remaining NaNs in label with 1 if they originated from fake replay entries
df_replay["label"] = df_replay["label"].fillna(1).astype(int)

# Combine and Save
df_merged = pd.concat([df_base, df_replay], ignore_index=True)
df_merged.to_csv(merged_csv_path, index=False)

print(f"Merge Complete! Saved to: {merged_csv_path}")
print("\nLabel Counts:")
print(df_merged["label"].value_counts(dropna=False))