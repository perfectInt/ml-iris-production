import logging

import joblib

from iris_classifier.config import MODEL_PATH, MODELS_DIR, RANDOM_STATE
from iris_classifier.data import load_dataset, split_dataset
from iris_classifier.model import build_model, evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train() -> float:
    """Обучает модель, сохраняет артефакт и возвращает test accuracy."""
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(features, target)

    model = build_model(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    accuracy = evaluate_model(model, x_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    logger.info("Модель сохранена: %s", MODEL_PATH)
    logger.info("Accuracy на тестовой выборке: %.4f", accuracy)

    return accuracy


if __name__ == "__main__":
    train()
