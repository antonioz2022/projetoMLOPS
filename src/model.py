import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_preprocessor(X_train):
    # Detecta tipos de colunas
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []

    # Pipeline para colunas numéricas
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    # Pipeline para colunas categóricas
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def build_models(X_train, y_train):
    preprocessor = build_preprocessor(X_train)

    # 🔥 CALCULA PESO AUTOMÁTICO (ESSENCIAL)
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)

    scale_pos_weight = neg / pos if pos > 0 else 1

    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.2f}")

    # =========================
    # XGBoost (modelo principal)
    # =========================
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # 🔥 CRUCIAL
        objective="binary:logistic",
        eval_metric="aucpr",  # melhor pra desbalanceado
        random_state=42,
        n_jobs=-1
    )

    # =========================
    # Random Forest (baseline)
    # =========================
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",  # já ajuda
        random_state=42,
        n_jobs=-1,
    )

    # Pipelines
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_model),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_model),
        ]
    )

    return {
        "xgboost": xgb_pipeline,
        "random_forest": rf_pipeline
    }