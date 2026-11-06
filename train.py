"""Воспроизводимое обучение MLP для лабораторной работы 0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import sklearn
import torch
import yaml
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RUNS_DIRECTORY = Path("runs")
SPLIT_PATH = Path("data/split.json")
MANIFEST_PATH = Path("data/manifest.json")


class RunPruned(Exception):
    """Запуск остановлен прунером Optuna."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def calculate_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Конфигурация должна быть словарём YAML")
    return config


def validate_config(config: dict[str, Any]) -> None:
    try:
        model = config["model"]
        train = config["train"]
        runtime = config["runtime"]
        seed = int(config["seed"])
        split_seed = int(config["split_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Неполная конфигурация: {error}") from error

    if split_seed != 0:
        raise ValueError("split_seed должен быть равен 0")
    if seed < 0:
        raise ValueError("seed не может быть отрицательным")
    if int(model["n_layers"]) not in {1, 2, 3}:
        raise ValueError("n_layers должен быть равен 1, 2 или 3")
    if int(model["hidden"]) <= 0:
        raise ValueError("hidden должен быть положительным")
    if not 0.0 <= float(model["dropout"]) <= 0.5:
        raise ValueError("dropout должен быть от 0 до 0.5")
    if float(train["lr"]) <= 0:
        raise ValueError("lr должен быть положительным")
    if float(train["weight_decay"]) < 0:
        raise ValueError("weight_decay не может быть отрицательным")
    if int(train["batch_size"]) <= 0:
        raise ValueError("batch_size должен быть положительным")
    if int(train["epochs"]) != 30:
        raise ValueError("В ЛР0 число эпох должно быть равно 30")
    if runtime["device"] != "cpu":
        raise ValueError("В этой работе используется device: cpu")
    if int(runtime["num_threads"]) <= 0:
        raise ValueError("num_threads должен быть положительным")


def git_information(require_clean: bool) -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if require_clean:
            raise RuntimeError(
                "Итоговый запуск возможен только внутри Git-репозитория"
            ) from error
        return "not-available", "not-available"

    if require_clean and dirty:
        raise RuntimeError(
            "Есть незакоммиченные изменения. "
            "Итоговый запуск остановлен:\n" + dirty
        )
    return commit, dirty


def set_determinism(seed: int, num_threads: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class MLP(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = 64
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(input_size, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_size = hidden
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def load_data() -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], str]:
    if not SPLIT_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Сначала выполните python prepare_data.py"
        )

    split_sha256 = calculate_sha256(SPLIT_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_sha256 = manifest["datasets"]["load_digits"]["split"]["sha256"]
    if split_sha256 != expected_sha256:
        raise ValueError("SHA-256 split.json не совпадает с манифестом")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    y = y.astype(np.int64)
    return X, y, split, split_sha256


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    indices: list[int],
) -> tuple[float, float, list[int]]:
    model.eval()
    inputs = torch.from_numpy(X[indices])
    targets = torch.from_numpy(y[indices])
    with torch.no_grad():
        logits = model(inputs)
        loss = criterion(logits, targets).item()
        predictions = logits.argmax(dim=1)
    accuracy = float((predictions == targets).float().mean().item())
    return loss, accuracy, predictions.tolist()


def append_history(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "epoch_time_seconds",
    ]
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "index", "y_true", "y_pred"],
        )
        writer.writeheader()
        writer.writerows(rows)


def metrics_from_files(
    history_path: Path,
    predictions_path: Path,
) -> dict[str, float]:
    with history_path.open("r", encoding="utf-8", newline="") as file:
        history = list(csv.DictReader(file))
    with predictions_path.open("r", encoding="utf-8", newline="") as file:
        predictions = list(csv.DictReader(file))

    metrics: dict[str, float] = {
        "val_loss": float(history[-1]["val_loss"]),
        "training_time_seconds": sum(
            float(row["epoch_time_seconds"]) for row in history
        ),
    }
    for split_name in ["val", "test"]:
        rows = [row for row in predictions if row["split"] == split_name]
        if rows:
            correct = sum(row["y_true"] == row["y_pred"] for row in rows)
            metrics[f"{split_name}_acc"] = correct / len(rows)
    return metrics


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in flatten_config(value).items():
                result[f"{key}.{nested_key}"] = nested_value
        else:
            result[key] = value
    return result


def run_experiment(
    config: dict[str, Any],
    *,
    series: str,
    evaluate_test: bool,
    final_run: bool,
    nested_mlflow: bool = False,
    prune_callback: Callable[[int, float], bool] | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    if evaluate_test and not final_run:
        raise ValueError("--eval-test разрешён только вместе с --final-run")

    commit, dirty = git_information(require_clean=final_run)
    X, y, split, split_sha256 = load_data()

    seed = int(config["seed"])
    num_threads = int(config["runtime"]["num_threads"])
    generator = set_determinism(seed, num_threads)

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run_directory = RUNS_DIRECTORY / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    config_path = run_directory / "config.yaml"
    metadata_path = run_directory / "metadata.json"
    history_path = run_directory / "history.csv"
    predictions_path = run_directory / "predictions.csv"
    metrics_path = run_directory / "metrics.json"

    with config_path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=True,
        )

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "experiment": "lr0",
        "series": series,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "commit": commit,
        "dirty_worktree": bool(dirty),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "mlflow": package_version("mlflow"),
            "optuna": package_version("optuna"),
            "pyyaml": package_version("PyYAML"),
        },
        "device": "cpu",
        "num_threads": num_threads,
        "seed": seed,
        "split_sha256": split_sha256,
        "mlflow_run_id": None,
    }
    write_json(metadata_path, metadata)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0")

    tags = {
        "experiment": "lr0",
        "series": series,
        "commit": commit,
        "seed": seed,
    }
    if extra_tags:
        tags.update(extra_tags)

    try:
        with mlflow.start_run(run_name=run_id, nested=nested_mlflow) as active_run:
            metadata["mlflow_run_id"] = active_run.info.run_id
            write_json(metadata_path, metadata)
            mlflow.log_params(flatten_config(config))
            mlflow.set_tags(tags)

            model_config = config["model"]
            train_config = config["train"]
            model = MLP(
                n_layers=int(model_config["n_layers"]),
                hidden=int(model_config["hidden"]),
                dropout=float(model_config["dropout"]),
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(train_config["lr"]),
                weight_decay=float(train_config["weight_decay"]),
            )

            train_indices = split["train"]
            train_dataset = TensorDataset(
                torch.from_numpy(X[train_indices]),
                torch.from_numpy(y[train_indices]),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(train_config["batch_size"]),
                shuffle=True,
                num_workers=0,
                generator=generator,
            )

            for epoch in range(1, int(train_config["epochs"]) + 1):
                model.train()
                total_loss = 0.0
                total_examples = 0
                started = time.perf_counter()

                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()
                    batch_size = targets.size(0)
                    total_loss += loss.item() * batch_size
                    total_examples += batch_size

                epoch_time = time.perf_counter() - started
                train_loss = total_loss / total_examples
                val_loss, val_acc, _ = evaluate(
                    model,
                    criterion,
                    X,
                    y,
                    split["val"],
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_seconds": epoch_time,
                }
                append_history(history_path, row)
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "epoch_time_seconds": epoch_time,
                    },
                    step=epoch,
                )

                if prune_callback and prune_callback(epoch, val_acc):
                    raise RunPruned(f"Запуск остановлен после эпохи {epoch}")

            prediction_rows: list[dict[str, Any]] = []
            _, _, val_predictions = evaluate(
                model,
                criterion,
                X,
                y,
                split["val"],
            )
            for index, prediction in zip(split["val"], val_predictions):
                prediction_rows.append(
                    {
                        "split": "val",
                        "index": index,
                        "y_true": int(y[index]),
                        "y_pred": prediction,
                    }
                )

            if evaluate_test:
                _, _, test_predictions = evaluate(
                    model,
                    criterion,
                    X,
                    y,
                    split["test"],
                )
                for index, prediction in zip(split["test"], test_predictions):
                    prediction_rows.append(
                        {
                            "split": "test",
                            "index": index,
                            "y_true": int(y[index]),
                            "y_pred": prediction,
                        }
                    )

            write_predictions(predictions_path, prediction_rows)
            metrics = metrics_from_files(history_path, predictions_path)
            write_json(metrics_path, metrics)
            mlflow.log_metrics(metrics)

            metadata["status"] = "completed"
            metadata["finished_at"] = utc_now()
            write_json(metadata_path, metadata)
            mlflow.log_artifacts(str(run_directory))
            mlflow.set_tag("state", "COMPLETE")
            return {
                "run_id": run_id,
                "run_directory": str(run_directory),
                "metrics": metrics,
            }

    except RunPruned:
        metadata["status"] = "pruned"
        metadata["finished_at"] = utc_now()
        write_json(metadata_path, metadata)
        if mlflow.active_run():
            mlflow.set_tag("state", "PRUNED")
            mlflow.log_artifacts(str(run_directory))
        raise
    except Exception as error:
        metadata["status"] = "failed"
        metadata["finished_at"] = utc_now()
        metadata["error"] = str(error)
        metadata["traceback"] = traceback.format_exc()
        write_json(metadata_path, metadata)
        if mlflow.active_run():
            mlflow.set_tag("state", "FAIL")
            mlflow.log_artifacts(str(run_directory))
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--series", default="debug")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--final-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    result = run_experiment(
        config,
        series=args.series,
        evaluate_test=args.eval_test,
        final_run=args.final_run,
    )
    print("Запуск завершён:", result["run_id"])
    print("Каталог:", result["run_directory"])
    print("Метрики:", result["metrics"])


if __name__ == "__main__":
    main()
