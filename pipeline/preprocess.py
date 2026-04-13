import sys
from pathlib import Path
# Adiciona a raiz do projeto ao path para poder importar o src
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data_loader import load_raw_data
from src.preprocessor import DataPreprocessor
from src.config import PROCESSED_DATA_PATH

def main():
    # Ler hiperparâmetros do params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    print("A iniciar stage: PREPROCESS")
    
    # Carregar dados brutos (usando a função que já existia no seu data_loader.py)
    df = load_raw_data()
    
    # Inicializar o preprocessor
    preprocessor = DataPreprocessor(PROCESSED_DATA_PATH)
    
    # Executar o processamento e guardar em disco
    preprocessor.process_and_save(
        df=df,
        target_col=params['data']['target_column'],
        drop_cols=params['data']['drop_columns'],
        test_size=params['train']['test_size'],
        random_state=params['train']['random_state']
    )

if __name__ == "__main__":
    main()