"""Анализ экспериментов ЛР0 и построение отчётных файлов."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import optuna
import pandas as pd


RUNS_DIRECTORY = Path("runs")
REPORTS_DIRECTORY = Path("reports")
FIGURES_DIRECTORY = REPORTS_DIRECTORY / "figures"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def read_runs() -> list[dict[str, Any]]:
    records = []
    for directory in RUNS_DIRECTORY.iterdir():
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        metrics_path = directory / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        records.append(
            {
                "directory": directory,
                "metadata": metadata,
                "metrics": metrics,
            }
        )
    return records


def latest_run(
    records: list[dict[str, Any]],
    series: str,
    seed: int,
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record["metadata"].get("status") == "completed"
        and record["metadata"].get("series") == series
        and record["metadata"].get("seed") == seed
    ]
    if not candidates:
        raise ValueError(f"Не найден запуск {series}, seed={seed}")
    return max(
        candidates,
        key=lambda record: record["metadata"]["started_at"],
    )


def analyze_experiment1(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for seed in range(1, 6):
        record = latest_run(records, "base_seeds", seed)
        rows.append(
            {
                "seed": seed,
                "run_id": record["metadata"]["run_id"],
                "val_acc": record["metrics"]["val_acc"],
                "training_time_seconds": record["metrics"][
                    "training_time_seconds"
                ],
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "minimum": float(frame["val_acc"].min()),
        "median": float(frame["val_acc"].median()),
        "maximum": float(frame["val_acc"].max()),
        "range": float(
            frame["val_acc"].max() - frame["val_acc"].min()
        ),
    }
    return frame, summary


def analyze_search(
    records: list[dict[str, Any]],
) -> tuple[optuna.Study, dict[str, Any]]:
    study = optuna.load_study(
        study_name="lr0_search",
        storage="sqlite:///optuna.db",
    )
    state_counts = Counter(trial.state.name for trial in study.trials)

    search_runs = [
        record
        for record in records
        if record["metadata"].get("series") == "optuna_search"
    ]
    actual_epochs = 0
    actual_training_seconds = 0.0
    estimated_saved_seconds = 0.0

    for record in search_runs:
        history_path = record["directory"] / "history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        epochs = len(history)
        epoch_times = history["epoch_time_seconds"].astype(float)
        actual_epochs += epochs
        actual_training_seconds += float(epoch_times.sum())
        if 0 < epochs < 30:
            estimated_saved_seconds += float(
                (30 - epochs) * epoch_times.mean()
            )

    maximum_epochs = 40 * 30
    saved_epochs = maximum_epochs - actual_epochs
    importance = optuna.importance.get_param_importances(study)

    summary = {
        "total_trials": len(study.trials),
        "states": dict(sorted(state_counts.items())),
        "maximum_epochs": maximum_epochs,
        "actual_epochs": actual_epochs,
        "saved_epochs": saved_epochs,
        "saved_epoch_fraction": saved_epochs / maximum_epochs,
        "actual_training_seconds": actual_training_seconds,
        "estimated_saved_training_seconds": estimated_saved_seconds,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "param_importance": importance,
    }
    return study, summary


def build_search_figures(
    study: optuna.Study,
    importance: dict[str, float],
) -> None:
    trials = sorted(study.trials, key=lambda trial: trial.number)
    complete_x = []
    complete_y = []
    pruned_x = []
    pruned_y = []
    best_so_far = []
    running_best = -math.inf

    for trial in trials:
        if trial.state.name == "COMPLETE" and trial.value is not None:
            complete_x.append(trial.number)
            complete_y.append(trial.value)
            running_best = max(running_best, trial.value)
        elif trial.state.name == "PRUNED" and trial.value is not None:
            pruned_x.append(trial.number)
            pruned_y.append(trial.value)
        best_so_far.append(
            running_best if math.isfinite(running_best) else math.nan
        )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(complete_x, complete_y, label="COMPLETE", s=42)
    axis.scatter(
        pruned_x,
        pruned_y,
        label="PRUNED (последняя эпоха)",
        marker="x",
        s=48,
    )
    axis.step(
        [trial.number for trial in trials],
        best_so_far,
        where="post",
        label="Лучшее значение к этому trial",
        linewidth=2,
    )
    axis.set_xlabel("Номер trial")
    axis.set_ylabel("Точность на val")
    axis.set_title("Поиск гиперпараметров Optuna")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIRECTORY / "trial_history.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    ordered = sorted(importance.items(), key=lambda item: item[1])
    names = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(names, values)
    axis.set_xlabel("Важность")
    axis.set_title("Важность гиперпараметров")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIRECTORY / "parameter_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def analyze_final(
    records: list[dict[str, Any]],
    base_spread: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    total_fixed = 0
    total_spoiled = 0

    for seed in range(101, 106):
        base = latest_run(records, "final_base", seed)
        best = latest_run(records, "final_best", seed)
        base_predictions = pd.read_csv(
            base["directory"] / "predictions.csv"
        )
        best_predictions = pd.read_csv(
            best["directory"] / "predictions.csv"
        )
        base_test = base_predictions[base_predictions["split"] == "test"]
        best_test = best_predictions[best_predictions["split"] == "test"]
        comparison = base_test.merge(
            best_test,
            on=["split", "index", "y_true"],
            suffixes=("_base", "_best"),
        )
        base_correct = comparison["y_pred_base"] == comparison["y_true"]
        best_correct = comparison["y_pred_best"] == comparison["y_true"]
        fixed = int((~base_correct & best_correct).sum())
        spoiled = int((base_correct & ~best_correct).sum())
        total_fixed += fixed
        total_spoiled += spoiled

        row = {
            "seed": seed,
            "base_run_id": base["metadata"]["run_id"],
            "best_run_id": best["metadata"]["run_id"],
            "base_val_acc": base["metrics"]["val_acc"],
            "best_val_acc": best["metrics"]["val_acc"],
            "delta_val": (
                best["metrics"]["val_acc"]
                - base["metrics"]["val_acc"]
            ),
            "base_test_acc": base["metrics"]["test_acc"],
            "best_test_acc": best["metrics"]["test_acc"],
            "delta_test": (
                best["metrics"]["test_acc"]
                - base["metrics"]["test_acc"]
            ),
            "fixed_test": fixed,
            "spoiled_test": spoiled,
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    test_deltas = frame["delta_test"].tolist()
    val_deltas = frame["delta_val"].tolist()
    summary = {
        "delta_val_values": val_deltas,
        "delta_val_median": statistics.median(val_deltas),
        "delta_val_minimum": min(val_deltas),
        "delta_val_maximum": max(val_deltas),
        "delta_test_values": test_deltas,
        "delta_test_median": statistics.median(test_deltas),
        "delta_test_minimum": min(test_deltas),
        "delta_test_maximum": max(test_deltas),
        "mean_best_val": float(frame["best_val_acc"].mean()),
        "mean_best_test": float(frame["best_test_acc"].mean()),
        "mean_base_val": float(frame["base_val_acc"].mean()),
        "mean_base_test": float(frame["base_test_acc"].mean()),
        "fixed_test_total": total_fixed,
        "spoiled_test_total": total_spoiled,
        "base_spread": base_spread,
        "stable_by_assignment_rule": all(
            delta > base_spread for delta in test_deltas
        ),
    }
    return frame, summary


def main() -> None:
    REPORTS_DIRECTORY.mkdir(exist_ok=True)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records = read_runs()

    experiment1, experiment1_summary = analyze_experiment1(records)
    study, search_summary = analyze_search(records)
    build_search_figures(study, search_summary["param_importance"])
    final, final_summary = analyze_final(
        records,
        experiment1_summary["range"],
    )

    experiment1.to_csv(
        REPORTS_DIRECTORY / "experiment1.csv",
        index=False,
    )
    final.to_csv(
        REPORTS_DIRECTORY / "final_comparison.csv",
        index=False,
    )
    summary = {
        "experiment1": experiment1_summary,
        "optuna": search_summary,
        "final": final_summary,
    }
    write_json(REPORTS_DIRECTORY / "summary.json", summary)

    print("Эксперимент 1:", experiment1_summary)
    print("Optuna:", search_summary)
    print("Финальное сравнение:", final_summary)
    print("Создано:")
    for path in sorted(REPORTS_DIRECTORY.rglob("*")):
        if path.is_file():
            print(" -", path)


if __name__ == "__main__":
    main()
