import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =========================
# LOAD ORIGINAL DATA
# =========================

fake_df = pd.read_csv("dataset/Fake.csv")
real_df = pd.read_csv("dataset/Real.csv")

# =========================
# CREATE FOLDERS
# =========================

os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/validation", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

# =========================
# SPLIT FAKE DATA
# =========================

fake_train, fake_temp = train_test_split(
    fake_df,
    test_size=0.30,
    random_state=42
)

fake_val, fake_test = train_test_split(
    fake_temp,
    test_size=0.50,
    random_state=42
)

# =========================
# SPLIT REAL DATA
# =========================

real_train, real_temp = train_test_split(
    real_df,
    test_size=0.30,
    random_state=42
)

real_val, real_test = train_test_split(
    real_temp,
    test_size=0.50,
    random_state=42
)

# =========================
# SAVE TRAIN
# =========================

fake_train.to_csv(
    "dataset/train/Fake.csv",
    index=False
)

real_train.to_csv(
    "dataset/train/Real.csv",
    index=False
)

# =========================
# SAVE VALIDATION
# =========================

fake_val.to_csv(
    "dataset/validation/Fake.csv",
    index=False
)

real_val.to_csv(
    "dataset/validation/Real.csv",
    index=False
)

# =========================
# SAVE TEST
# =========================

fake_test.to_csv(
    "dataset/test/Fake.csv",
    index=False
)

real_test.to_csv(
    "dataset/test/Real.csv",
    index=False
)

print("✅ Dataset split completed")