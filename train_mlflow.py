import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score


def main(n_estimators: int, random_state: int):
    mlflow.set_experiment("iris-mlops")
    with mlflow.start_run():
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.sklearn.log_model(model, name="RandomForestModel")

        print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train iris RF and log with MLflow")
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args.n_estimators, args.random_state)
