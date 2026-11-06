# Лабораторная работа 0

Воспроизводимый эксперимент: подбор гиперпараметров MLP на наборе `load_digits`.

## Главный вопрос

Дал ли подбор гиперпараметров улучшение, которое больше случайного разброса?

Краткий результат: медианная разность на test составила +2,50 п.п., но один из пяти seed дал ухудшение -1,11 п.п. По строгому критерию работы данных недостаточно, чтобы назвать улучшение устойчивым. Подробности находятся в `reports/report.md`.

## Структура

```text
configs/              конфигурации базовой и лучшей моделей, пространство поиска
data/                 фиксированное разбиение и манифест
notebooks/            подготовительный Colab-блокнот
reports/              отчётные таблицы, рисунки и отчёт
prepare_data.py       подготовка и проверка данных
train.py              обучение, каталог запуска и MLflow
search.py             последовательный поиск Optuna
analyze.py            пересчёт таблиц и построение рисунков
requirements.txt      точные версии зависимостей
/configs/prediction.md         прогноз, зафиксированный до экспериментов
```

Каталоги `runs/`, `mlruns/` и базы `mlflow.db`, `optuna.db` не входят в Git и передаются отдельно как артефакты.

## Окружение

Проверенное окружение: Python 3.13.16, CPU, один поток.

```bash
python -m venv .venv
```

Активация в Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

## Подготовка данных

```bash
python prepare_data.py
```

Ожидаемые размеры: 1077/360/360. Ожидаемый SHA-256 `data/split.json`:

```text
d3ba070d3282c94e67fcf0f0bd7c10e0efc97dbcb00c07397313e14d38b5931a
```

## Обычный запуск

Отладочный запуск без доступа к test:

```bash
python train.py --config configs/base.yaml --series debug
```

Итоговый запуск требует чистого Git-дерева:

```bash
python train.py --config configs/base.yaml --series base_seeds --seed 1 --final-run
```

Доступ к test разрешён только с `--final-run`:

```bash
python train.py --config configs/best.yaml --series final_best --seed 101 --eval-test --final-run
```

Каждый запуск создаёт `runs/<run_id>/` с файлами `config.yaml`, `metadata.json`, `history.csv`, `predictions.csv` и `metrics.json`, а также регистрируется в MLflow.

## Optuna

Первые 10 trial:

```bash
python search.py --target-trials 10
```

Продолжение той же study до общего числа 40 trial:

```bash
python search.py --target-trials 40
```

Study хранится в `optuna.db`. Trial выполняются последовательно с `TPESampler(seed=42)` и `MedianPruner(n_startup_trials=5, n_warmup_steps=5)`.

## Анализ

```bash
python analyze.py
```

Скрипт создаёт CSV-таблицы, `reports/summary.json` и два PNG-рисунка. Числа отчёта не вводятся вручную.

## Артефакты

Для проверки передаются отдельно:

- каталоги запусков экспериментов 1 и 3 и trial поиска;
- `mlflow.db` и `mlruns/`;
- `optuna.db`;
- итоговая резервная ZIP-копия артефактов.
