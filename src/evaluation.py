from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]
    y_proba = model.predict_proba(X_test)[:, 1]
    
    threshold = 0.0753  # 🔥 TESTA 0.1–0.3
    
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    return y_prob, y_pred, cm