"""
Лабораторная работа №4: Ансамбли и полносвязные нейронные сети.
Реализация контрактной функции run_ensemble_experiment и серии экспериментов.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Импорт контрактов проверяющей системы (не удалять и не менять grader_contracts)
try:
    import grader_contracts
    from grader_contracts import EnsembleInput
    # Поддержка возможного названия выходного контракта в grader_contracts
    if hasattr(grader_contracts, "EnsembleOutput"):
        from grader_contracts import EnsembleOutput as ReturnContract
    elif hasattr(grader_contracts, "EnsembleResult"):
        from grader_contracts import EnsembleResult as ReturnContract
    elif hasattr(grader_contracts, "ExperimentResult"):
        from grader_contracts import ExperimentResult as ReturnContract
    else:
        ReturnContract = Any
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class EnsembleInput:
        train_features: pd.DataFrame
        train_target: pd.Series | np.ndarray
        test_features: pd.DataFrame
        test_target: pd.Series | np.ndarray

    @dataclass
    class ReturnContract:
        model_name: str
        test_probabilities: np.ndarray
        test_predictions: np.ndarray
        roc_auc: float
        accuracy: float
        precision: float
        recall: float
        f1: float


RANDOM_STATE = 42


def _extract_split(data: EnsembleInput):
    """Безопасное извлечение train/test выборок из входного контракта EnsembleInput."""
    X_train = getattr(data, "train_features", getattr(data, "X_train", None))
    y_train = getattr(data, "train_target", getattr(data, "y_train", None))
    X_test = getattr(data, "test_features", getattr(data, "X_test", None))
    y_test = getattr(data, "test_target", getattr(data, "y_test", None))

    X_train_df = pd.DataFrame(X_train).copy()
    X_test_df = pd.DataFrame(X_test).copy()

    # Выравнивание категориальных признаков (если есть строковые столбцы) только по train
    X_train_enc = pd.get_dummies(X_train_df, drop_first=True)
    X_test_enc = pd.get_dummies(X_test_df, drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    #Заполнение возможных пропусков медианами, вычисленными строго на train
    medians = X_train_enc.median(numeric_only=True)
    X_train_enc = X_train_enc.fillna(medians).fillna(0.0)
    X_test_enc = X_test_enc.fillna(medians).fillna(0.0)

    y_tr = np.asarray(y_train).ravel()
    y_te = np.asarray(y_test).ravel()

    # Приведение меток {1, 2} -> {0, 1}, если необходимо
    if set(np.unique(y_tr)) == {1, 2}:
        y_tr = y_tr - 1
        y_te = y_te - 1

    return X_train_enc, y_tr, X_test_enc, y_te


def _build_return_object(
    model_name: str,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    roc_auc: float,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> ReturnContract:
    """Формирование объекта возврата в точном соответствии с полями контракта."""
    if ReturnContract is Any or not inspect.isclass(ReturnContract):
        return {
            "model_name": model_name,
            "test_probabilities": y_prob,
            "test_predictions": y_pred,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    sig = inspect.signature(ReturnContract)
    kwargs: dict[str, Any] = {}

    for param_name in sig.parameters:
        low = param_name.lower()
        if "name" in low or low == "model":
            kwargs[param_name] = model_name
        elif "prob" in low or "score" in low and "auc" not in low and "f1" not in low:
            kwargs[param_name] = y_prob
        elif "pred" in low or low == "predictions":
            kwargs[param_name] = y_pred
        elif "auc" in low or "roc" in low:
            kwargs[param_name] = roc_auc
        elif "acc" in low:
            kwargs[param_name] = accuracy
        elif "prec" in low:
            kwargs[param_name] = precision
        elif "rec" in low:
            kwargs[param_name] = recall
        elif "f1" in low:
            kwargs[param_name] = f1

    return ReturnContract(**kwargs)


def build_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Словарь исследуемых моделей (ансамбли и полносвязная нейронная сеть)."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        class_weight=None,
        random_state=random_state,
        n_jobs=1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=random_state,
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=random_state,
    )

    ada = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=random_state,
    )

    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="tanh",
                    alpha=0.1,
                    learning_rate_init=0.01,
                    max_iter=500,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    stacking = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("gb", gb),
            ("mlp", mlp),
        ],
        final_estimator=LogisticRegression(C=1.0, random_state=random_state),
        cv=5,
        n_jobs=1,
    )

    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("gb", gb),
            ("mlp", mlp),
        ],
        voting="soft",
        weights=[3, 2, 2, 1],
        n_jobs=1,
    )

    return {
        "RandomForest (Tuned)": rf,
        "ExtraTrees (Tuned)": et,
        "GradientBoosting (Tuned)": gb,
        "HistGradientBoosting (Tuned)": hgb,
        "AdaBoost (Tuned)": ada,
        "MLPClassifier (Tuned)": mlp,
        "StackingClassifier (RF+ET+GB+MLP)": stacking,
        "SoftVotingEnsemble (RF+ET+GB+MLP)": voting,
    }


def run_ensemble_experiment(data: EnsembleInput) -> ReturnContract:
    """
    Контрактная функция лабораторной работы №4.
    Обучает выбранный ансамбль строго на train-выборке и оценивает метрики на test-выборке.
    """
    X_train, y_train, X_test, y_test = _extract_split(data)

    # Выбранная финальная модель: Soft Voting ансамбль разнородных моделей (RF + ET + GB + MLP)
    models = build_candidate_models(random_state=RANDOM_STATE)
    model_name = "SoftVotingEnsemble (RF+ET+GB+MLP)"
    model = models[model_name]

    # Обучение строго на тренировочной выборке
    model.fit(X_train, y_train)

    # Предсказание вероятностей положительного класса и бинарных меток на тесте
    y_prob = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
    y_pred = np.asarray(model.predict(X_test), dtype=int)

    # Вычисление метрик на тестовой выборке
    roc_auc = float(roc_auc_score(y_test, y_prob))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    return _build_return_object(
        model_name=model_name,
        y_prob=y_prob,
        y_pred=y_pred,
        roc_auc=roc_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def run_all_benchmarks(csv_path: str | Path = "german.csv") -> dict[str, Any]:
    """Запуск всех экспериментов на german.csv и сохранение experiment_results.json."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    target_col = "Creditability" if "Creditability" in df.columns else df.columns[0]

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    if set(y.unique()) == {1, 2}:
        y = y - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = build_candidate_models(random_state=RANDOM_STATE)
    experiments_log = []

    for name, clf in models.items():
        clf.fit(X_train, y_train)

        prob_train = clf.predict_proba(X_train)[:, 1]
        prob_test = clf.predict_proba(X_test)[:, 1]
        pred_test = clf.predict(X_test)

        metrics = {
            "model_name": name,
            "train_roc_auc": round(float(roc_auc_score(y_train, prob_train)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob_test)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
            "precision": round(float(precision_score(y_test, pred_test, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred_test, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred_test, zero_division=0)), 4),
        }
        experiments_log.append(metrics)

    # Проверка контрактной функции
    contract_input = EnsembleInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
    )
    final_res = run_ensemble_experiment(contract_input)

    selected_summary = next(
        item for item in experiments_log if item["model_name"] == "SoftVotingEnsemble (RF+ET+GB+MLP)"
    )

    payload = {
        "dataset": "german.csv",
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "selected_model": selected_summary,
        "experiments": sorted(experiments_log, key=lambda x: x["roc_auc"], reverse=True),
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Эксперименты успешно завершены. Результаты сохранены в experiment_results.json:\n")
    summary_df = pd.DataFrame(payload["experiments"])
    print(summary_df.to_string(index=False))
    return payload


if __name__ == "__main__":
    run_all_benchmarks("german.csv")