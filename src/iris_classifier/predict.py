import argparse

import joblib
import pandas as pd

from iris_classifier.config import MODEL_PATH


def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> int:
    """Загружает модель и предсказывает класс одного объекта."""
    if not MODEL_PATH.exists():
        message = (
            f"Файл модели не найден: {MODEL_PATH}. "
            "Сначала запустите обучение: poetry run python -m iris_classifier.train"
        )
        raise FileNotFoundError(message)

    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )

    return int(model.predict(features)[0])


def parse_args() -> argparse.Namespace:
    """Читает параметры объекта из командной строки."""
    parser = argparse.ArgumentParser(description="Предсказание класса Iris.")
    parser.add_argument("--sepal-length", type=float, required=True)
    parser.add_argument("--sepal-width", type=float, required=True)
    parser.add_argument("--petal-length", type=float, required=True)
    parser.add_argument("--petal-width", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prediction = predict(
        sepal_length=args.sepal_length,
        sepal_width=args.sepal_width,
        petal_length=args.petal_length,
        petal_width=args.petal_width,
    )

    print(f"Предсказанный класс: {prediction}")
