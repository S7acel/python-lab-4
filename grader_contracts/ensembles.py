from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EnsembleInput:
    train_features: Any
    train_target: Any
    test_features: Any
    test_target: Any
    random_seed: int = 42

@dataclass(frozen=True)
class EnsembleResult:
    model_name: str
    probabilities: Any
    predictions: Any
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
