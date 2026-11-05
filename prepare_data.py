"""Подготовка воспроизводимого разбиения данных для ЛР0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


SPLIT_SEED = 0
DATA_DIRECTORY = Path("data")
SPLIT_PATH = DATA_DIRECTORY / "split.json"
MANIFEST_PATH = DATA_DIRECTORY / "manifest.json"


def write_json(path: Path, data: dict) -> None:
    """Записать JSON в стабильном формате."""

    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def calculate_sha256(path: Path) -> str:
    """Вычислить SHA-256 файла."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_split(
    y: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Проверить обязательные свойства разбиения."""

    expected_sizes = {
        "train": 1077,
        "val": 360,
        "test": 360,
    }

    actual_sizes = {
        "train": len(train_indices),
        "val": len(val_indices),
        "test": len(test_indices),
    }

    if actual_sizes != expected_sizes:
        raise ValueError(
            f"Неверные размеры частей: {actual_sizes}"
        )

    train_set = set(int(index) for index in train_indices)
    val_set = set(int(index) for index in val_indices)
    test_set = set(int(index) for index in test_indices)

    if train_set & val_set:
        raise ValueError("train и val пересекаются")

    if train_set & test_set:
        raise ValueError("train и test пересекаются")

    if val_set & test_set:
        raise ValueError("val и test пересекаются")

    combined = train_set | val_set | test_set

    if combined != set(range(len(y))):
        raise ValueError(
            "Части не покрывают все записи набора"
        )

    expected_classes = set(range(10))

    for name, indices in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        actual_classes = set(
            int(value) for value in np.unique(y[indices])
        )

        if actual_classes != expected_classes:
            raise ValueError(
                f"В части {name} отсутствуют классы"
            )


def main() -> None:
    """Создать split.json и manifest.json."""

    X, y = load_digits(return_X_y=True)
    all_indices = np.arange(len(y))

    rest_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=y,
        random_state=SPLIT_SEED,
    )

    train_indices, val_indices = train_test_split(
        rest_indices,
        test_size=0.25,
        stratify=y[rest_indices],
        random_state=SPLIT_SEED,
    )

    validate_split(
        y,
        train_indices,
        val_indices,
        test_indices,
    )

    DATA_DIRECTORY.mkdir(exist_ok=True)

    split_data = {
        "dataset": "sklearn.datasets.load_digits",
        "split_seed": SPLIT_SEED,
        "train": sorted(
            int(index) for index in train_indices
        ),
        "val": sorted(
            int(index) for index in val_indices
        ),
        "test": sorted(
            int(index) for index in test_indices
        ),
    }

    write_json(SPLIT_PATH, split_data)
    split_sha256 = calculate_sha256(SPLIT_PATH)

    manifest_data = {
        "schema_version": 1,
        "datasets": {
            "load_digits": {
                "source": "sklearn.datasets.load_digits",
                "package": "scikit-learn",
                "package_version": sklearn.__version__,
                "total_records": int(len(y)),
                "features_per_record": int(X.shape[1]),
                "classes": [
                    int(value) for value in np.unique(y)
                ],
                "feature_value_range": [
                    float(X.min()),
                    float(X.max()),
                ],
                "selection": {
                    "included_records": int(len(y)),
                    "excluded_records": 0,
                    "rule": (
                        "Использованы все записи набора "
                        "load_digits без исключений"
                    ),
                },
                "split": {
                    "file": "data/split.json",
                    "sha256": split_sha256,
                    "split_seed": SPLIT_SEED,
                    "rule": (
                        "Стратифицированное разбиение: "
                        "сначала 20% в test, затем "
                        "25% остатка в val"
                    ),
                    "sizes": {
                        "train": int(len(train_indices)),
                        "val": int(len(val_indices)),
                        "test": int(len(test_indices)),
                    },
                },
            }
        },
    }

    write_json(MANIFEST_PATH, manifest_data)

    print("Подготовка данных завершена")
    print("train:", len(train_indices))
    print("val:", len(val_indices))
    print("test:", len(test_indices))
    print("SHA-256 split.json:", split_sha256)


if __name__ == "__main__":
    main()
