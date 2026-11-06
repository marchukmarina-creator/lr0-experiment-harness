"""Последовательный поиск гиперпараметров Optuna для ЛР0."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml
from optuna.study import MaxTrialsCallback

from train import RunPruned, load_config, run_experiment


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Файл {path} должен содержать словарь YAML")
    return data


def create_objective(
    base_config: dict[str, Any],
    search_config: dict[str, Any],
):
    space = search_config["search_space"]

    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)
        config["seed"] = int(search_config["fixed"]["seed"])
        config["split_seed"] = int(
            search_config["fixed"]["split_seed"]
        )
        config["train"]["epochs"] = int(
            search_config["fixed"]["epochs"]
        )

        config["train"]["lr"] = trial.suggest_float(
            "lr",
            float(space["lr"]["low"]),
            float(space["lr"]["high"]),
            log=bool(space["lr"]["log"]),
        )
        config["model"]["n_layers"] = trial.suggest_int(
            "n_layers",
            int(space["n_layers"]["low"]),
            int(space["n_layers"]["high"]),
        )
        config["model"]["hidden"] = trial.suggest_int(
            "hidden",
            int(space["hidden"]["low"]),
            int(space["hidden"]["high"]),
            log=bool(space["hidden"]["log"]),
        )
        config["model"]["dropout"] = trial.suggest_float(
            "dropout",
            float(space["dropout"]["low"]),
            float(space["dropout"]["high"]),
        )
        config["train"]["weight_decay"] = trial.suggest_float(
            "weight_decay",
            float(space["weight_decay"]["low"]),
            float(space["weight_decay"]["high"]),
            log=bool(space["weight_decay"]["log"]),
        )
        config["train"]["batch_size"] = trial.suggest_categorical(
            "batch_size",
            list(space["batch_size"]["choices"]),
        )

        def prune_callback(epoch: int, val_acc: float) -> bool:
            trial.report(val_acc, step=epoch)
            return trial.should_prune()

        try:
            result = run_experiment(
                config,
                series="optuna_search",
                evaluate_test=False,
                final_run=True,
                nested_mlflow=True,
                prune_callback=prune_callback,
                extra_tags={
                    "trial_number": trial.number,
                    "trial_state": "RUNNING",
                },
            )
        except RunPruned as error:
            trial.set_user_attr("prune_reason", str(error))
            raise optuna.TrialPruned(str(error)) from error

        trial.set_user_attr("run_id", result["run_id"])
        value = float(result["metrics"]["val_acc"])
        return value

    return objective


def print_summary(study: optuna.Study) -> None:
    states = Counter(trial.state.name for trial in study.trials)
    print("Всего trial:", len(study.trials))
    print("Состояния:", dict(sorted(states.items())))

    if study.best_trial is not None:
        print("Лучший trial:", study.best_trial.number)
        print("Лучшая val_acc:", study.best_value)
        print("Лучшие параметры:", study.best_params)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path("configs/search_space.yaml"),
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        required=True,
        help="Общее целевое число trial в study",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_config = load_config(args.base_config)
    search_config = load_yaml(args.search_config)
    study_config = search_config["study"]

    if args.target_trials <= 0:
        raise ValueError("target-trials должен быть положительным")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0")

    study = optuna.create_study(
        study_name=str(study_config["name"]),
        storage=str(study_config["storage"]),
        load_if_exists=True,
        direction=str(study_config["direction"]),
        sampler=optuna.samplers.TPESampler(
            seed=int(study_config["sampler_seed"])
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=int(
                study_config["pruner"]["n_startup_trials"]
            ),
            n_warmup_steps=int(
                study_config["pruner"]["n_warmup_steps"]
            ),
        ),
    )

    existing = len(study.trials)
    if existing >= args.target_trials:
        print(
            f"В study уже {existing} trial; "
            f"цель {args.target_trials} достигнута"
        )
        print_summary(study)
        return

    parent_run_id = study.user_attrs.get("mlflow_parent_run_id")
    if parent_run_id:
        parent_context = mlflow.start_run(run_id=parent_run_id)
    else:
        parent_context = mlflow.start_run(run_name="lr0_search")

    with parent_context as parent_run:
        if not parent_run_id:
            study.set_user_attr(
                "mlflow_parent_run_id",
                parent_run.info.run_id,
            )

        mlflow.set_tags(
            {
                "experiment": "lr0",
                "series": "optuna_search",
                "study_name": study.study_name,
            }
        )

        study.optimize(
            create_objective(base_config, search_config),
            n_trials=args.target_trials,
            n_jobs=1,
            callbacks=[
                MaxTrialsCallback(
                    args.target_trials,
                    states=None,
                )
            ],
        )

    print_summary(study)


if __name__ == "__main__":
    main()
