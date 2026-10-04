from iris_classifier.data import load_dataset, split_dataset
from iris_classifier.model import build_model, evaluate_model


def test_dataset_has_expected_shape() -> None:
    features, target = load_dataset()

    assert features.shape == (150, 4)
    assert target.shape == (150,)


def test_model_produces_valid_accuracy() -> None:
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(features, target)

    model = build_model()
    model.fit(x_train, y_train)

    accuracy = evaluate_model(model, x_test, y_test)

    assert 0.0 <= accuracy <= 1.0
    assert accuracy >= 0.8
