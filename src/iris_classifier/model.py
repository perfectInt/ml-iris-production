import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

type Model = Pipeline


def build_model(random_state: int = 42) -> Model:
    """Создаёт ML-пайплайн для обучения классификатора."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ],
    )


def evaluate_model(
    model: Model,
    features: pd.DataFrame,
    target: pd.Series,
) -> float:
    """Возвращает accuracy модели на переданных данных."""
    predictions = model.predict(features)
    return float(accuracy_score(target, predictions))
