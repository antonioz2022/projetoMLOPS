import pandas as pd
from .config import RAW_DATA_PATH

def load_raw_data():
    # Agora apontamos diretamente para o arquivo CSV dentro da pasta
    file_path = RAW_DATA_PATH / "Digital_Payment_Fraud_Detection_Dataset.csv"
    
    print(f"Lendo dados de: {file_path}")
    df = pd.read_csv(file_path)

    print("Raw dataset:", df.shape)

    return df