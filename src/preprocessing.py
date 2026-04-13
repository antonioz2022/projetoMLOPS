import pandas as pd
import numpy as np
from .config import PROCESSED_DATA_PATH



def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # =========================
    # Diagnóstico inicial
    # =========================

    missing = df.isna().sum().sort_values(ascending=False)
    duplicates = df.duplicated().sum()

    print(f"Linhas duplicadas: {duplicates}")
    print("Colunas com valores ausentes:")
    print(missing[missing > 0])

    df_clean = df.copy()

    # =========================
    # Remoção de duplicatas
    # =========================

    before = len(df_clean)

    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    after = len(df_clean)

    print(f"Duplicatas removidas: {before - after}")

    # =========================
    # Padronização de strings
    # =========================

    str_cols = df_clean.select_dtypes(include="object").columns.tolist()

    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # =========================
    # Conversão de colunas numéricas
    # =========================

    expected_numeric = [
        "transaction_amount",
        "account_age_days",
        "transaction_hour",
        "previous_failed_attempts",
        "avg_transaction_amount",
        "is_international",
        "ip_risk_score",
        "login_attempts_last_24h",
        "fraud_label",
    ]

    for col in expected_numeric:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    print("Limpeza inicial concluída.")

    # =========================
    # Função auxiliar de validação
    # =========================

    def clip_if_exists(frame, column, low=None, high=None):
        if column in frame.columns:
            frame[column] = frame[column].clip(lower=low, upper=high)

    # =========================
    # Validações de faixa
    # =========================

    # Hora da transação
    clip_if_exists(df_clean, "transaction_hour", 0, 23)

    # Score de risco IP
    clip_if_exists(df_clean, "ip_risk_score", 0, 1)

    # =========================
    # Colunas binárias
    # =========================

    for col in ["is_international", "fraud_label"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(int).clip(0, 1)

    # =========================
    # Valores inválidos
    # =========================

    for col in ["transaction_amount", "avg_transaction_amount"]:
        if col in df_clean.columns:

            invalid = (df_clean[col] <= 0).sum()

            if invalid > 0:
                print(f"{col}: {invalid} valores não positivos convertidos em NaN")

                df_clean.loc[df_clean[col] <= 0, col] = np.nan

    # account_age_days negativo

    if "account_age_days" in df_clean.columns:

        negatives = (df_clean["account_age_days"] < 0).sum()

        if negatives > 0:

            print(f"account_age_days: {negatives} valores negativos convertidos em NaN")

            df_clean.loc[df_clean["account_age_days"] < 0, "account_age_days"] = np.nan

    print("Validações concluídas.")

    return df_clean


def save_processed(df):

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Processed dataset salvo em:", PROCESSED_DATA_PATH)