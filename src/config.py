from pathlib import Path

RAW_DATA_PATH = Path("data/raw/Digital_Payment_Fraud_Detection_Dataset.csv")

PROCESSED_DATA_PATH = Path("data/processed/clean_transactions.csv")

MODEL_PATH = Path("models/model.pkl")

TARGET = "fraud_label"

DROP_COLUMNS = [
    "transaction_id",
    "user_id"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42