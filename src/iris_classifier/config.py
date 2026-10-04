from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "iris_classifier.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
