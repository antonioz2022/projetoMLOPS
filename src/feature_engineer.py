import pandas as pd


def create_features(df: pd.DataFrame):

    df_feat = df.copy()

    # Razão entre valor da transação e média histórica
    if {"transaction_amount", "avg_transaction_amount"}.issubset(df_feat.columns):

        df_feat["amount_vs_avg"] = (
            df_feat["transaction_amount"]
            / (df_feat["avg_transaction_amount"] + 1)
        )

        df_feat["high_value_transaction"] = (
            df_feat["transaction_amount"]
            > 2 * df_feat["avg_transaction_amount"]
        ).astype(int)

    # Transação noturna
    if "transaction_hour" in df_feat.columns:

        df_feat["night_transaction"] = (
            df_feat["transaction_hour"]
            .between(0, 5)
            .astype(int)
        )

    # Score comportamental
    if {"login_attempts_last_24h", "previous_failed_attempts"}.issubset(df_feat.columns):

        df_feat["behavior_risk_score"] = (
            df_feat["login_attempts_last_24h"]
            + df_feat["previous_failed_attempts"]
        )

    # Conta nova
    if "account_age_days" in df_feat.columns:

        df_feat["new_account"] = (
            df_feat["account_age_days"] < 30
        ).astype(int)

    # Internacional + risco alto de IP
    if {"is_international", "ip_risk_score"}.issubset(df_feat.columns):

        df_feat["international_high_risk"] = (
            (df_feat["is_international"] == 1)
            & (df_feat["ip_risk_score"] > 0.70)
        ).astype(int)

    # Atividade suspeita
    if {
        "login_attempts_last_24h",
        "transaction_amount",
        "avg_transaction_amount",
    }.issubset(df_feat.columns):

        df_feat["high_activity_risk"] = (
            (df_feat["login_attempts_last_24h"] > 5)
            & (df_feat["transaction_amount"] > df_feat["avg_transaction_amount"])
        ).astype(int)

    print("Shape após feature engineering:", df_feat.shape)

    return df_feat