from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.modeling import fit_best_model_and_evaluate, run_validation_experiments
from src.preprocessing import (
    BASE_RAW_COLUMNS,
    LEAKAGE_COLUMNS,
    OPTIONAL_RAW_COLUMNS,
    POST_PUBLICATION_COLUMNS,
    load_raw_dataset,
    prepare_features,
    split_dataset,
)


def _log(message: str) -> None:
    print(f"[CP1] {message}", flush=True)


def _save_figures(df: pd.DataFrame, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    sample = df.sample(n=min(20_000, len(df)), random_state=42).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(sample["price"], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Price distribution")
    axes[0].set_xlabel("Price")
    sns.histplot(sample["price"], bins=50, ax=axes[1], color="#55A868", log_scale=(False, True))
    axes[1].set_title("Price distribution (log y-scale)")
    axes[1].set_xlabel("Price")
    fig.tight_layout()
    fig.savefig(images_dir / "price_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    top_cities = sample["city"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_cities.index, top_cities.values, color="#C44E52")
    ax.set_title("Top 15 cities by listing count")
    ax.set_xlabel("Listings")
    fig.tight_layout()
    fig.savefig(images_dir / "top_cities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="accommodates",
        y="price",
        hue="room_type",
        alpha=0.35,
        s=20,
        ax=ax,
    )
    ax.set_title("Accommodates vs price")
    ax.set_ylim(0, sample["price"].quantile(0.99))
    fig.tight_layout()
    fig.savefig(images_dir / "accommodates_vs_price.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_summary(df: pd.DataFrame, validation_results: pd.DataFrame, test_result: dict[str, float], output_dir: Path) -> None:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": "price",
        "target_mean": float(df["price"].mean()),
        "target_median": float(df["price"].median()),
        "target_q95": float(df["price"].quantile(0.95)),
        "target_q99": float(df["price"].quantile(0.99)),
        "missing_top_10": (df.isna().mean().sort_values(ascending=False).head(10) * 100).round(2).to_dict(),
        "zero_price_rows": int((df["price"] == 0).sum()),
        "price_eq_1000_rows": int((df["price"] == 1000).sum()),
        "excluded_leakage_features": LEAKAGE_COLUMNS,
        "excluded_post_publication_features_for_final_model": POST_PUBLICATION_COLUMNS,
        "validation_results": validation_results.to_dict(orient="records"),
        "final_test_result": test_result,
    }
    (output_dir / "cp1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def run_cp1(train_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path("report/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Старт пайплайна CP1. train_path={train_path}")
    _log("Шаг 1/6. Загружаю полный датасет для EDA и сводки.")
    raw_full_df = pd.read_csv(
        train_path,
        usecols=list(dict.fromkeys(BASE_RAW_COLUMNS + OPTIONAL_RAW_COLUMNS + LEAKAGE_COLUMNS)),
        low_memory=False,
    )
    _log(
        "Датасет загружен: "
        f"{raw_full_df.shape[0]:,} строк, {raw_full_df.shape[1]} колонок, "
        f"средняя цена={raw_full_df['price'].mean():.2f}, медиана={raw_full_df['price'].median():.2f}"
    )

    _log("Шаг 2/6. Строю графики для CP1.")
    _save_figures(raw_full_df, images_dir)
    _log(f"Графики сохранены в {images_dir}")

    _log("Шаг 3/6. Готовлю признаки для моделирования без постфактум-признаков.")
    raw_model_df = load_raw_dataset(train_path, include_post_publication_features=False)
    X, y = prepare_features(raw_model_df, include_post_publication_features=False)
    _log(f"Подготовлена матрица признаков: {X.shape[0]:,} строк, {X.shape[1]} признаков")
    split = split_dataset(X, y)
    _log(
        "Сплит готов: "
        f"train={len(split.X_train):,}, validation={len(split.X_val):,}, test={len(split.X_test):,}"
    )

    _log("Шаг 4/6. Запускаю валидационные эксперименты.")
    validation_results, best_model_name = run_validation_experiments(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
    )
    for row in validation_results.itertuples(index=False):
        _log(
            "Validation result: "
            f"{row.model_name} -> MAE={row.mae:.3f}, RMSE={row.rmse:.3f}, R2={row.r2:.3f}"
        )
    _log(f"Лучшая модель на validation: {best_model_name}")

    _log("Шаг 5/6. Дообучаю лучшую модель на train+validation и считаю test-метрики.")
    _, test_result = fit_best_model_and_evaluate(
        model_name=best_model_name,
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        X_test=split.X_test,
        y_test=split.y_test,
    )
    _log(
        "Test result: "
        f"{test_result.model_name} -> MAE={test_result.mae:.3f}, "
        f"RMSE={test_result.rmse:.3f}, R2={test_result.r2:.3f}"
    )

    _log("Шаг 6/6. Сохраняю артефакты CP1 в report/cp1.")
    validation_results.to_csv(output_dir / "cp1_validation_results.csv", index=False)
    test_result_payload = {
        "model_name": test_result.model_name,
        "mae": round(test_result.mae, 3),
        "rmse": round(test_result.rmse, 3),
        "r2": round(test_result.r2, 3),
    }
    (output_dir / "cp1_test_result.json").write_text(json.dumps(test_result_payload, indent=2))
    _build_summary(raw_full_df, validation_results, test_result_payload, output_dir)
    _log(f"Готово. Артефакты сохранены в {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP1 pipeline for the Airbnb price prediction project.")
    parser.add_argument("--train-path", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/cp1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cp1(train_path=args.train_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
