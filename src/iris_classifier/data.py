from typing import TypeAlias

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from iris_classifier.config import RANDOM_STATE, TEST_SIZE

DatasetSplit: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Загружает встроенный датасет Iris в табличном формате."""
    iris = load_iris(as_frame=True)

    features = iris.data
    target = iris.target

    return features, target


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Делит данные на обучающую и тестовую выборки."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
