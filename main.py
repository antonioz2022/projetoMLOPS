from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data
from src.preprocessing import clean_data, save_processed
from src.feature_engineer import create_features
from src.model import build_models
from src.evaluation import evaluate_model
from src.config import TARGET, DROP_COLUMNS, TEST_SIZE, RANDOM_STATE


def main():
    df = load_raw_data()

    df = clean_data(df)

    df = create_features(df)

    save_processed(df)

    X = df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = build_models(X_train, y_train)

    for model_name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Treinando modelo: {model_name}")
        print(f"{'=' * 50}")

        model.fit(X_train, y_train)

        print(f"\nAvaliação do modelo: {model_name}")
        y_prob, y_pred, cm = evaluate_model(model, X_test, y_test)

        print(cm)


if __name__ == "__main__":
    main()