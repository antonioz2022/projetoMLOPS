import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import save_npz, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    """Classe para processar os dados, criar features e aplicar transformações."""
    
    def __init__(self, processed_path: Path):
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.preprocessor = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.drop_duplicates().reset_index(drop=True)
        
        # Padronização de strings
        str_cols = df_clean.select_dtypes(include="object").columns.tolist()
        for col in str_cols:
            df_clean[col] = df_clean[col].astype(str).str.strip()

        # Correção de numéricos e limites
        for col in ["transaction_amount", "avg_transaction_amount"]:
            if col in df_clean.columns:
                df_clean.loc[df_clean[col] <= 0, col] = np.nan

        if "transaction_hour" in df_clean.columns:
            df_clean["transaction_hour"] = df_clean["transaction_hour"].clip(0, 23)
            
        if "ip_risk_score" in df_clean.columns:
            df_clean["ip_risk_score"] = df_clean["ip_risk_score"].clip(0, 1)

        for col in ["is_international", "fraud_label"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0).astype(int).clip(0, 1)

        return df_clean

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = df.copy()
        
        if {"transaction_amount", "avg_transaction_amount"}.issubset(df_feat.columns):
            df_feat["amount_vs_avg"] = df_feat["transaction_amount"] / (df_feat["avg_transaction_amount"] + 1)
            df_feat["high_value_transaction"] = (df_feat["transaction_amount"] > 2 * df_feat["avg_transaction_amount"]).astype(int)

        if "transaction_hour" in df_feat.columns:
            df_feat["night_transaction"] = df_feat["transaction_hour"].between(0, 5).astype(int)

        if {"login_attempts_last_24h", "previous_failed_attempts"}.issubset(df_feat.columns):
            df_feat["behavior_risk_score"] = df_feat["login_attempts_last_24h"] + df_feat["previous_failed_attempts"]

        if "account_age_days" in df_feat.columns:
            df_feat["new_account"] = (df_feat["account_age_days"] < 30).astype(int)

        if {"is_international", "ip_risk_score"}.issubset(df_feat.columns):
            df_feat["international_high_risk"] = ((df_feat["is_international"] == 1) & (df_feat["ip_risk_score"] > 0.70)).astype(int)

        return df_feat

    def build_transformer(self, X_train):
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if numeric_features:
            numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
            transformers.append(("num", numeric_transformer, numeric_features))

        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            transformers.append(("cat", categorical_transformer, categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)
        return self.preprocessor

    def process_and_save(self, df: pd.DataFrame, target_col: str, drop_cols: list, test_size: float, random_state: int):
        print("Iniciando limpeza e feature engineering...")
        df_clean = self.clean_data(df)
        df_feat = self.create_features(df_clean)

        X = df_feat.drop(columns=[target_col] + drop_cols, errors="ignore")
        y = df_feat[target_col].values

        print("Separando Treino e Teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print("Ajustando preprocessor...")
        self.build_transformer(X_train)
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        print("Salvando matrizes e preprocessor em disco...")
        
        # Converte os arrays normais para matrizes esparsas
        X_train_sparse = csr_matrix(X_train_transformed)
        X_test_sparse = csr_matrix(X_test_transformed)

        # Guarda as matrizes convertidas
        save_npz(self.processed_path / "X_train.npz", X_train_sparse)
        save_npz(self.processed_path / "X_test.npz", X_test_sparse)
        
        # O resto mantém-se igual
        np.save(self.processed_path / "y_train.npy", y_train)
        np.save(self.processed_path / "y_test.npy", y_test)
        joblib.dump(self.preprocessor, self.processed_path / "preprocessor.joblib")
        
        print("Processamento concluído com sucesso!")