[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# Предсказание цены краткосрочной аренды жилья по характеристикам объявления

**Студент:** Мартыновских Мирра Максимовна

**Группа:** БИВ232

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Данные](#данные)
3. [Структура репозитория](#структура-репозитория)
4. [Быстрый старт](#быстрый-старт)
5. [Результаты CP1](#результаты-cp1)
6. [Отчёт](#отчёт)

## Описание задачи

Цель проекта: построить модель машинного обучения, которая предсказывает цену краткосрочной аренды жилья по параметрам объявления Airbnb.

- **Тип задачи:** регрессия
- **Целевая переменная:** `price`
- **Источник данных:** [Airbnb Property Rental Price (Kaggle)](https://www.kaggle.com/datasets/datavidia/airbnb-property-rental-price)
- **Размер датасета:** 261,894 объявлений, 55 колонок в `train.csv`
- **Основная метрика:** MAE
- **Дополнительные метрики:** RMSE, R²

Почему основной метрикой выбрана `MAE`:
- цена имеет тяжелый хвост и заметное число дорогих объектов;
- MAE проще интерпретировать в деньгах;
- RMSE оставляем как дополнительную метрику, чтобы видеть чувствительность к большим ошибкам.

## Данные

В проекте используются файлы:

- `data/raw/train.csv` — тренировочная выборка с таргетом `price`
- `data/raw/test.csv` — тестовая выборка без таргета
- `data/raw/sample_submission.csv` — пример формата сабмита

Что сделано на этапе CP1:

- проверены размеры и типы данных;
- проверено отсутствие полных дублей и дублей по `id`;
- обработаны пропуски в числовых и категориальных признаках;
- добавлены простые engineered features:
  - `amenities_count`
  - `host_verifications_count`
  - `bathrooms_text_num`
  - `name_word_count`
  - `description_word_count`
  - `host_tenure_days`
- исключен явный признак-утечка `estimated_revenue_l365d`;
- признаки с постфактум-информацией о поведении пользователей и отзывах не использовались в финальной модели CP1.

Сплит для экспериментов:

- `train`: 70%
- `validation`: 15%
- `test`: 15%

## Структура репозитория

```text
.
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
├── presentation
├── report
│   ├── cp1
│   │   ├── cp1_summary.json
│   │   ├── cp1_test_result.json
│   │   └── cp1_validation_results.csv
│   ├── images
│   │   ├── accommodates_vs_price.png
│   │   ├── price_distribution.png
│   │   └── top_cities.png
│   └── report.md
├── src
│   ├── __init__.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── run_cp1.py
├── tests
│   ├── conftest.py
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Быстрый старт

### Локальный запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.run_cp1
```

### Запуск через Docker Compose

```bash
docker compose up --build cp1
```

### Запуск через Docker без Compose

```bash
docker build -t airbnb-cp1 .
docker run --rm -v "$(pwd)":/app airbnb-cp1
```

Что будет создано после запуска:

- `report/cp1/cp1_validation_results.csv`
- `report/cp1/cp1_test_result.json`
- `report/cp1/cp1_summary.json`
- графики в `report/images/`

Проверка качества кода:

```bash
ruff check src tests --line-length 120
pytest -q
```

## Результаты CP1

Ключевые наблюдения по данным:

- в обучающей выборке нет дублей строк и дублей `id`;
- медианная цена равна `136`, средняя — `207.08`;
- 95-й перцентиль цены — `680.35`, 99-й — `950`;
- есть только одно нулевое значение цены;
- значение `1000` встречается 1183 раза, что похоже на верхний cap в данных.

Результаты на validation:

| Модель | MAE | RMSE | R² | Комментарий |
|--------|-----:|-----:|---:|-------------|
| Dummy median | 125.10 | 209.66 | -0.13 | Базовая точка отсчета |
| Ridge + OHE | 73.63 | 320.37 | -1.64 | Линейная модель слабо описывает распределение |
| RandomForest (50k train sample) | 68.03 | 119.21 | 0.63 | Рабочий нелинейный baseline |
| HistGradientBoosting | **66.98** | **116.88** | **0.65** | Лучшая модель CP1 |

Финальный результат лучшей модели на test:

| Модель | MAE | RMSE | R² |
|--------|-----:|-----:|---:|
| HistGradientBoosting | **65.99** | **113.76** | **0.66** |

Почему в CP1 выбрана именно `HistGradientBoosting`:

- лучшее качество на validation;
- устойчивее линейной модели на тяжелом хвосте цен;
- использует только признаки, доступные из самого объявления и профиля хоста;
- не опирается на явные утечки вроде `estimated_revenue_l365d`.

Артефакты CP1:

- [validation results](report/cp1/cp1_validation_results.csv)
- [test result](report/cp1/cp1_test_result.json)
- [summary](report/cp1/cp1_summary.json)

## Отчёт

Черновик отчёта находится в [`report/report.md`](report/report.md).
