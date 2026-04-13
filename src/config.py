import os
from dotenv import load_dotenv
from pathlib import Path

# Carregar .env
load_dotenv()

# Caminhos baseados na raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"
MLFLOW_PATH = PROJECT_ROOT / "mlruns"
METRICS_PATH = PROJECT_ROOT / "metrics"

# Criar diretórios se não existirem
for path in [DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH, LOGS_PATH, METRICS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Variáveis de ambiente e MLFlow
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detection_model")
ENV = os.getenv("ENV", "development")
DEBUG = os.getenv("DEBUG", "True") == "True"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{MLFLOW_PATH}")