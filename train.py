from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# Параметры модели
N_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Данные
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Создаём MLflow эксперимент
mlflow.set_experiment("breast_cancer_classification")

with mlflow.start_run():
    # Обучение
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Метрики
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Логируем параметры
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)

    # Логируем метрики
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Логируем модель как артефакт
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Регистрируем в Model Registry
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, "BreastCancerRandomForest")

    print(f"Accuracy: {accuracy:.2%}, F1: {f1:.4f}")
    print(f"Run ID: {run_id}")

# Сохраняем локально для FastAPI
joblib.dump(model, "model.joblib")
print("model.joblib saved!")