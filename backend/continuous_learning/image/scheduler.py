import os
import time
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from dataset_builder import build_training_dataset, TRAINING_DATASET, REPLAY_DATASET
from retrain_rf import load_training_dataset, prepare_data, split_dataset, train_random_forest, save_model, NEW_MODEL
from evaluate_models import load_models, CURRENT_MODEL

RETRAIN_THRESHOLD = 5  # Minimum approved samples needed to trigger training cycle

def evaluate_and_promote():
    """Compares current production model vs newly trained RF_v2.pkl."""
    if not os.path.exists(TRAINING_DATASET) or not os.path.exists(NEW_MODEL):
        print("[Scheduler] Artifacts missing for model evaluation.")
        return

    df = pd.read_csv(TRAINING_DATASET)
    X, y = prepare_data(df)

    try:
        current_model, new_model = load_models()
        acc_current = accuracy_score(y, current_model.predict(X))
        acc_new = accuracy_score(y, new_model.predict(X))

        print("\n" + "=" * 50)
        print(f"[Evaluation Summary]")
        print(f"Active Model Accuracy : {acc_current:.4f}")
        print(f"New Model Accuracy    : {acc_new:.4f}")
        print("=" * 50)

        if acc_new >= acc_current:
            print("🚀 New Model is better/equal! Updating production model...")
            joblib.dump(new_model, CURRENT_MODEL)
            print("✓ Production model swapped successfully.")
        else:
            print("⚡ Active model performed better. Keeping existing model.")
    except Exception as e:
        print(f"[Evaluation Error]: {e}")

def run_learning_cycle():
    """Main workflow check executed on schedule."""
    print("\n[Scheduler] Checking replay dataset for approved feedback...")
    if not os.path.exists(REPLAY_DATASET):
        print("[Scheduler] replay_dataset.csv does not exist yet.")
        return

    df = pd.read_csv(REPLAY_DATASET)
    approved_samples = df[df["status"] == "Approved"]
    print(f"[Scheduler] Found {len(approved_samples)} Approved samples.")

    if len(approved_samples) < RETRAIN_THRESHOLD:
        print(f"[Scheduler] Need at least {RETRAIN_THRESHOLD} approved samples. Waiting for user feedback...")
        return

    print("\n--- STEP 1: Building Training Dataset ---")
    build_training_dataset()

    print("\n--- STEP 2: Retraining Random Forest Model ---")
    train_df = load_training_dataset()
    X, y = prepare_data(train_df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = train_random_forest(X_train, y_train)
    save_model(model)

    print("\n--- STEP 3: Evaluating & Promoting Best Model ---")
    evaluate_and_promote()

if __name__ == "__main__":
    print("🤖 Continuous Learning Scheduler Active.")
    while True:
        run_learning_cycle()
        time.sleep(30)  # Checks every 30 seconds