# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# =========================
# SETTINGS
# =========================

INPUT_CANDIDATES = [
    "Матрица данных по пшенице.xlsx"
]
FALLBACK_PREPARED_CANDIDATES = [
    Path("results_sarimax_extended_exog") / "sarimax_extended_exog_results.xlsx",
    Path("sarimax_extended_exog_results.xlsx"),
]

TRAIN_END = pd.Timestamp("2024-12-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-01")
MIN_TRAIN_OBS = 18
USE_ACTUAL_FUTURE_EXOG = False

CATBOOST_PARAMS = dict(iterations=250, depth=2, learning_rate=0.03, l2_leaf_reg=10, random_seed=42)
RANDOM_FOREST_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_leaf=3, random_state=42)

EXOG_BLOCKS = {
    "market_regulation": ["world_price", "usd_rub", "duty"],
    "cost_macro_logistics": ["duty", "diesel", "cpi", "freight_tariff"],
    "full_economic": ["duty", "diesel", "cpi", "freight_tariff", "world_price", "usd_rub"],
    "duty_only": ["duty"],
}

DROP_FROM_EXCEL = {
    "iterations", "depth", "learning_rate", "l2_leaf_reg", "random_seed",
    "n_estimations", "n_estimators", "max_depth", "min_samples_leaf", "random_stare",
    "random_state", "setting",
}

OUTPUT_FOLDER = "results_ml_fixed_hyperparams_walkforward_with_importance_no_lag12"
OUTPUT_FILE = "ml_fixed_hyperparams_walkforward_results_with_importance_no_lag12.xlsx"

# =========================
# UTILITIES
# =========================


def norm_col(x: Any) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def unique(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def find_file(candidates: list[Any]) -> Path | None:
    for folder in [Path.cwd(), Path(__file__).resolve().parent]:
        for name in candidates:
            p = folder / name
            if p.exists():
                return p
    return None


def make_output_path() -> Path:
    for base in [Path(__file__).resolve().parent, Path.cwd(), Path.home() / "Documents", Path.home()]:
        folder = base / OUTPUT_FOLDER
        try:
            if folder.exists() and not folder.is_dir():
                folder = folder.with_name(f"{folder.name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
            folder.mkdir(parents=True, exist_ok=True)
            test = folder / "__write_test.tmp"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return folder / OUTPUT_FILE
        except Exception:
            pass
    raise RuntimeError("Не удалось создать папку для результатов.")


def detect_date_target(df: pd.DataFrame) -> tuple[str, str]:
    names = {c: norm_col(c) for c in df.columns}
    date_cols = [c for c, n in names.items() if n in {"date", "дата", "period", "месяц"}]
    if not date_cols:
        date_cols = [c for c in df.columns if pd.to_datetime(df[c], errors="coerce").notna().sum() >= max(10, len(df) * 0.6)]
    if not date_cols:
        raise ValueError("Не найден столбец с датой.")

    target_cols = [c for c, n in names.items() if n in {"target", "price", "цена", "цена_пшеницы", "producer_price", "y"}]
    if not target_cols:
        numeric = [
            c for c in df.columns if c != date_cols[0]
            and pd.to_numeric(df[c], errors="coerce").notna().sum() >= max(10, len(df) * 0.6)
        ]
        if not numeric:
            raise ValueError("Не найден столбец с целевой переменной.")
        target_cols = [numeric[0]]
    return date_cols[0], target_cols[0]


def finalize_data(df: pd.DataFrame) -> pd.DataFrame:
    date_col, target_col = ("date", "target") if {"date", "target"}.issubset(df.columns) else detect_date_target(df)
    df = df.rename(columns={date_col: "date", target_col: "target", **{c: norm_col(c) for c in df.columns if c not in {date_col, target_col}}})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["date", "target"]).sort_values("date").reset_index(drop=True)

    month = df["date"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["lag1"] = df["target"].shift(1)
    return df


def read_matrix(path: Path) -> pd.DataFrame:
    return finalize_data(pd.read_excel(path, sheet_name=0).dropna(how="all"))


def read_prepared(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = next((s for s in ["Prepared_data", "prepared_data", "Data", "data"] if s in xls.sheet_names), xls.sheet_names[0])
    return finalize_data(pd.read_excel(path, sheet_name=sheet).dropna(how="all").rename(columns=lambda c: norm_col(c)))


def load_data() -> tuple[pd.DataFrame, str]:
    required_exog = sorted({f for block in EXOG_BLOCKS.values() for f in block})
    main = find_file(INPUT_CANDIDATES)
    fallback = find_file(FALLBACK_PREPARED_CANDIDATES)

    if main:
        df = read_matrix(main)
        missing = [c for c in required_exog if c not in df.columns]
        if not missing:
            source = f"Основная матрица: {main}"
        elif fallback:
            df = read_prepared(fallback)
            source = f"Основная матрица без exog {missing}; использован Prepared_data: {fallback}"
        else:
            raise FileNotFoundError(f"В основной матрице нет exog {missing}, резервный файл не найден.")
    elif fallback:
        df = read_prepared(fallback)
        source = f"Prepared_data: {fallback}"
    else:
        raise FileNotFoundError("Не найдена матрица данных или резервный sarimax_extended_exog_results.xlsx.")

    for c in required_exog:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["date"] <= TEST_END].copy(), source

# =========================
# FEATURES AND MODELS
# =========================


def build_feature_sets() -> dict[str, list[str]]:
    fs = {
        "baseline_season_no_lag": ["month_sin", "month_cos"],
        "baseline_lag1_season": ["lag1", "month_sin", "month_cos"],
    }
    for name, block in EXOG_BLOCKS.items():
        fs[f"{name}_season_no_lag"] = unique(["month_sin", "month_cos", *block])
        fs[f"{name}_lag1_season"] = unique(["lag1", "month_sin", "month_cos", *block])
    return fs


FEATURE_SETS = build_feature_sets()


def model_names() -> list[str]:
    names = ["RandomForest"]
    if CATBOOST_AVAILABLE:
        names.insert(0, "CatBoost")
    else:
        print("[WARN] CatBoost не установлен: будет рассчитан только RandomForest.")
    return names


def make_model(name: str):
    if name == "CatBoost":
        return CatBoostRegressor(**CATBOOST_PARAMS, loss_function="RMSE", verbose=False, allow_writing_files=False)
    if name == "RandomForest":
        return RandomForestRegressor(**RANDOM_FOREST_PARAMS, n_jobs=-1)
    raise ValueError(name)


def model_params(name: str) -> dict[str, Any]:
    return CATBOOST_PARAMS.copy() if name == "CatBoost" else RANDOM_FOREST_PARAMS.copy()


def calc_metrics(y_true, y_pred) -> dict[str, float]:

    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_pct": np.nan, "R2": np.nan}

    nonzero = y_true != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if np.any(nonzero) else np.nan

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE_pct": mape,
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
    }


def target_history(df: pd.DataFrame, end_date: pd.Timestamp) -> dict[pd.Timestamp, float]:
    part = df[df["date"] <= end_date].dropna(subset=["target"])
    return {pd.Timestamp(r.date): float(r.target) for r in part.itertuples(index=False)}


def feature_row(df: pd.DataFrame, d: pd.Timestamp, features: list[str], known_y: dict[pd.Timestamp, float], exog_date: pd.Timestamp) -> dict[str, float]:
    row, month = {}, float(d.month)
    for f in features:
        if f == "month_sin":
            row[f] = float(np.sin(2 * np.pi * month / 12))
        elif f == "month_cos":
            row[f] = float(np.cos(2 * np.pi * month / 12))
        elif f == "lag1":
            row[f] = float(known_y[d - pd.DateOffset(months=1)])
        else:
            if USE_ACTUAL_FUTURE_EXOG:
                s = df.loc[df["date"] == d, f]
                if len(s) == 1 and pd.notna(s.iloc[0]):
                    row[f] = float(s.iloc[0]); continue
            row[f] = float(df.loc[df["date"] == exog_date, f].iloc[0])
    return row


def fit_predict(model_name: str, train: pd.DataFrame, features: list[str], xrow: dict[str, float]) -> tuple[Any, float]:
    model = make_model(model_name)
    model.fit(train[features], train["target"])
    pred = float(model.predict(pd.DataFrame([xrow], columns=features))[0])
    return model, pred

# =========================
# FORECASTS
# =========================


def walk_forward(df: pd.DataFrame, model_name: str, feature_set: str, features: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for d in pd.date_range(TEST_START, TEST_END, freq="MS"):
        train = df[df["date"] < d].dropna(subset=features + ["target"])
        actual = df.loc[df["date"] == d, "target"]
        if len(train) < MIN_TRAIN_OBS or len(actual) != 1 or pd.isna(actual.iloc[0]):
            continue
        train_start = train["date"].min()
        last = train["date"].max()
        xrow = feature_row(df, d, features, target_history(df, last), last)
        _, pred = fit_predict(model_name, train, features, xrow)
        rows.append(
            make_forecast_record(
                "walk_forward_2025_one_step",
                model_name,
                feature_set,
                features,
                d,
                actual.iloc[0],
                pred,
                xrow,
                extra={
                    "fold_date": d,
                    "n_train_obs": len(train),
                    "train_start": train_start,
                    "train_end": last,
                },
            )
        )
    out = pd.DataFrame(rows)
    return out, metrics_or_nan(out)


def recursive_forecast(df: pd.DataFrame, model_name: str, feature_set: str, features: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    train = df[df["date"] <= TRAIN_END].dropna(subset=features + ["target"])
    if len(train) < MIN_TRAIN_OBS:
        raise ValueError("Слишком мало наблюдений в train после удаления NA.")

    model = make_model(model_name)
    model.fit(train[features], train["target"])
    known_y, rows = target_history(df, TRAIN_END), []

    train_start = train["date"].min()
    train_end = train["date"].max()

    for d in pd.date_range(TEST_START, TEST_END, freq="MS"):
        xrow = feature_row(df, d, features, known_y, TRAIN_END)
        pred = float(model.predict(pd.DataFrame([xrow], columns=features))[0])
        known_y[d] = pred
        actual = df.loc[df["date"] == d, "target"]
        actual = float(actual.iloc[0]) if len(actual) == 1 and pd.notna(actual.iloc[0]) else np.nan
        rows.append(
            make_forecast_record(
                "recursive_annual_2025",
                model_name,
                feature_set,
                features,
                d,
                actual,
                pred,
                xrow,
                extra={
                    "n_train_obs": len(train),
                    "train_start": train_start,
                    "train_end": train_end,
                },
            )
        )
    out = pd.DataFrame(rows)
    return out, metrics_or_nan(out)


def make_forecast_record(
    mode: str,
    model: str,
    feature_set: str,
    features: list[str],
    d,
    actual,
    forecast,
    xrow: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Формирует строку прогноза и сразу добавляет базовые ошибки."""
    err = actual - forecast if not pd.isna(actual) else np.nan
    abs_error = abs(err) if not pd.isna(err) else np.nan
    squared_error = err ** 2 if not pd.isna(err) else np.nan

    if not pd.isna(actual) and float(actual) != 0:
        ape_pct = abs((actual - forecast) / actual) * 100
    else:
        ape_pct = np.nan

    row = {
        "mode": mode,
        "model": model,
        "feature_set": feature_set,
        "features": ", ".join(features),
        "date": d,
        "actual": actual,
        "forecast": forecast,
        "error": err,
        "abs_error": abs_error,
        "squared_error": squared_error,
        "APE_pct": ape_pct,
        **xrow,
    }
    if extra:
        row.update(extra)
    return row


def metrics_or_nan(forecast_df: pd.DataFrame) -> dict[str, float]:
    ok = forecast_df.dropna(subset=["actual", "forecast"])
    return calc_metrics(ok["actual"], ok["forecast"]) if not ok.empty else {"MAE": np.nan, "RMSE": np.nan, "MAPE_pct": np.nan, "R2": np.nan}


def compute_wf_fold_metrics(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    if forecasts_df is None or forecasts_df.empty:
        return pd.DataFrame()

    wf = forecasts_df[forecasts_df["mode"].eq("walk_forward_2025_one_step")].copy()
    if wf.empty:
        return pd.DataFrame()

    if "fold_date" not in wf.columns:
        wf["fold_date"] = wf["date"]

    group_cols = ["mode", "model", "feature_set", "features", "fold_date"]
    rows: list[dict[str, Any]] = []

    for keys, g in wf.groupby(group_cols, dropna=False):
        ok = g.dropna(subset=["actual", "forecast"])
        metrics = metrics_or_nan(ok)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_obs_in_fold": len(ok),
            "actual_mean": ok["actual"].mean() if not ok.empty else np.nan,
            "forecast_mean": ok["forecast"].mean() if not ok.empty else np.nan,
            "error_mean": (ok["actual"] - ok["forecast"]).mean() if not ok.empty else np.nan,
            "abs_error_mean": np.abs(ok["actual"] - ok["forecast"]).mean() if not ok.empty else np.nan,
            "squared_error_mean": np.square(ok["actual"] - ok["forecast"]).mean() if not ok.empty else np.nan,
            **metrics,
        })

        for col in ["n_train_obs", "train_start", "train_end"]:
            if col in g.columns:
                row[col] = g[col].iloc[0]

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model", "feature_set", "fold_date"]).reset_index(drop=True)
    return out


def summarize_wf_fold_metrics(wf_fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Сводит помесячные WF-метрики по каждой модели и спецификации."""
    if wf_fold_metrics_df is None or wf_fold_metrics_df.empty:
        return pd.DataFrame()

    out = (
        wf_fold_metrics_df
        .groupby(["mode", "model", "feature_set", "features"], as_index=False)
        .agg(
            n_folds=("fold_date", "nunique"),
            fold_start=("fold_date", "min"),
            fold_end=("fold_date", "max"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAE_min=("MAE", "min"),
            MAE_max=("MAE", "max"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            RMSE_min=("RMSE", "min"),
            RMSE_max=("RMSE", "max"),
            MAPE_pct_mean=("MAPE_pct", "mean"),
            MAPE_pct_std=("MAPE_pct", "std"),
            MAPE_pct_min=("MAPE_pct", "min"),
            MAPE_pct_max=("MAPE_pct", "max"),
            n_train_obs_min=("n_train_obs", "min") if "n_train_obs" in wf_fold_metrics_df.columns else ("n_obs_in_fold", "min"),
            n_train_obs_max=("n_train_obs", "max") if "n_train_obs" in wf_fold_metrics_df.columns else ("n_obs_in_fold", "max"),
        )
        .sort_values(["MAPE_pct_mean", "RMSE_mean", "MAE_mean"], na_position="last")
        .reset_index(drop=True)
    )
    out.insert(0, "rank_by_WF_fold_MAPE_mean", np.arange(1, len(out) + 1))
    return out

# =========================
# FEATURE IMPORTANCE
# =========================


def feature_importance(model, model_name: str, features: list[str]) -> pd.DataFrame:
    vals = np.asarray(model.get_feature_importance() if model_name == "CatBoost" else model.feature_importances_, dtype=float)
    total = float(np.nansum(vals))
    pct = vals / total * 100 if total > 0 else np.full_like(vals, np.nan, dtype=float)
    return pd.DataFrame({"feature": features, "importance_raw": vals, "importance_pct": pct})


def importance_records(mode: str, model_name: str, feature_set: str, features: list[str], model, n_train: int, fold_date=None) -> list[dict[str, Any]]:
    imp = feature_importance(model, model_name, features).sort_values("importance_pct", ascending=False, na_position="last")
    rows = []
    for rank, r in enumerate(imp.itertuples(index=False), start=1):
        row = {
            "mode": mode, "model": model_name, "feature_set": feature_set, "features": ", ".join(features),
            "feature_rank": rank, "feature": r.feature, "importance_raw": r.importance_raw,
            "importance_pct": r.importance_pct, "n_train_obs": n_train,
        }
        if fold_date is not None:
            row["fold_date"] = fold_date
        rows.append(row)
    return rows


def compute_importance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rec_rows, fold_rows = [], []
    for model_name in model_names():
        for fs_name, feats in FEATURE_SETS.items():
            if missing_features(df, feats):
                continue

            train = df[df["date"] <= TRAIN_END].dropna(subset=feats + ["target"])
            if len(train) >= MIN_TRAIN_OBS:
                model = make_model(model_name)
                model.fit(train[feats], train["target"])
                rec_rows += importance_records("recursive_annual_2025_train_until_2024_12", model_name, fs_name, feats, model, len(train))

            for d in pd.date_range(TEST_START, TEST_END, freq="MS"):
                train = df[df["date"] < d].dropna(subset=feats + ["target"])
                if len(train) < MIN_TRAIN_OBS:
                    continue
                model = make_model(model_name)
                model.fit(train[feats], train["target"])
                fold_rows += importance_records("walk_forward_2025_one_step", model_name, fs_name, feats, model, len(train), d)

    rec_df, folds_df = pd.DataFrame(rec_rows), pd.DataFrame(fold_rows)
    mean_df = summarize_fold_importance(folds_df)
    return rec_df, mean_df, folds_df


def summarize_fold_importance(folds_df: pd.DataFrame) -> pd.DataFrame:
    if folds_df.empty:
        return pd.DataFrame()
    mean_df = (
        folds_df.groupby(["mode", "model", "feature_set", "features", "feature"], as_index=False)
        .agg(
            importance_pct_mean=("importance_pct", "mean"),
            importance_pct_std=("importance_pct", "std"),
            importance_raw_mean=("importance_raw", "mean"),
            importance_raw_std=("importance_raw", "std"),
            n_folds=("fold_date", "nunique"),
            n_train_obs_min=("n_train_obs", "min"),
            n_train_obs_max=("n_train_obs", "max"),
        )
        .sort_values(["model", "feature_set", "importance_pct_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    mean_df["feature_rank"] = mean_df.groupby(["model", "feature_set"]).cumcount() + 1
    first = ["mode", "model", "feature_set", "features", "feature_rank", "feature"]
    return mean_df[first + [c for c in mean_df.columns if c not in first]]

# =========================
# RUN AND EXPORT
# =========================


def missing_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f not in df.columns]


def evaluate_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, forecasts = [], []
    for model_name in model_names():
        for fs_name, feats in FEATURE_SETS.items():
            miss = missing_features(df, feats)
            if miss:
                print(f"[WARN] Пропущен {model_name} / {fs_name}: нет {miss}")
                continue
            print(f"Расчёт: {model_name} | {fs_name}")
            try:
                wf_df, wf_m = walk_forward(df, model_name, fs_name, feats)
                rec_df, rec_m = recursive_forecast(df, model_name, fs_name, feats)
                forecasts += [x for x in [wf_df, rec_df] if not x.empty]
                summary.append({
                    "model": model_name, "feature_set": fs_name, "features": ", ".join(feats), **model_params(model_name),
                    **{f"WF2025_{k}": v for k, v in wf_m.items()},
                    **{f"REC2025_{k}": v for k, v in rec_m.items()},
                })
            except Exception as e:
                summary.append({"model": model_name, "feature_set": fs_name, "features": ", ".join(feats), **model_params(model_name), "error": str(e)})

    summary_df = pd.DataFrame(summary)
    if not summary_df.empty and "REC2025_MAPE_pct" in summary_df:
        summary_df = summary_df.sort_values(["REC2025_MAPE_pct", "WF2025_MAPE_pct", "REC2025_RMSE"], na_position="last").reset_index(drop=True)
        summary_df.insert(0, "rank_by_REC2025_mape", np.arange(1, len(summary_df) + 1))
    return summary_df, pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_FROM_EXCEL if c in df.columns], errors="ignore") if df is not None and not df.empty else df


def settings(source: str) -> pd.DataFrame:
    return pd.DataFrame([
        ("DATA_SOURCE", source),
        ("TRAIN_END", str(TRAIN_END.date())),
        ("TEST_PERIOD", f"{TEST_START.date()} — {TEST_END.date()}"),
        ("USE_ACTUAL_FUTURE_EXOG", USE_ACTUAL_FUTURE_EXOG),
        ("CATBOOST_PARAMS", str(CATBOOST_PARAMS)),
        ("RANDOM_FOREST_PARAMS", str(RANDOM_FOREST_PARAMS)),
        ("EXOG_BLOCKS", str(EXOG_BLOCKS)),
        ("FEATURE_SETS", str(FEATURE_SETS)),
        ("NOTE", "Гиперпараметры фиксированы и одинаковы для walk-forward и рекурсивного прогноза."),
        ("NOTE_WF_FOLDS", "Листы WF_fold_metrics и WF_fold_metrics_summary содержат метрики по каждому fold/month walk-forward и их сводку."),
        ("NOTE_EXOG", "При USE_ACTUAL_FUTURE_EXOG=False будущие exog фиксируются на последнем доступном уровне."),
    ], columns=["parameter", "value"])


def main() -> None:
    df, source = load_data()
    print(f"Источник данных: {source}")
    print(f"Период: {df['date'].min().date()} — {df['date'].max().date()} | строк: {len(df)}")
    print(f"CatBoost доступен: {CATBOOST_AVAILABLE}\n")

    summary_df, forecasts_df = evaluate_all(df)
    wf_fold_metrics_df = compute_wf_fold_metrics(forecasts_df)
    wf_fold_metrics_summary_df = summarize_wf_fold_metrics(wf_fold_metrics_df)
    fi_rec_df, fi_wf_mean_df, fi_wf_folds_df = compute_importance(df)

    out = make_output_path()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        clean(summary_df).to_excel(writer, sheet_name="Summary_metrics", index=False)
        clean(forecasts_df).to_excel(writer, sheet_name="Forecasts_long", index=False)
        clean(wf_fold_metrics_df).to_excel(writer, sheet_name="WF_fold_metrics", index=False)
        clean(wf_fold_metrics_summary_df).to_excel(writer, sheet_name="WF_fold_metrics_summary", index=False)
        clean(fi_rec_df).to_excel(writer, sheet_name="Feature_importance_REC", index=False)
        clean(fi_wf_mean_df).to_excel(writer, sheet_name="Feature_importance_WF_mean", index=False)
        clean(fi_wf_folds_df).to_excel(writer, sheet_name="Feature_importance_WF_folds", index=False)
        clean(settings(source)).to_excel(writer, sheet_name="Settings", index=False)

    print(f"\nГотово. Результаты сохранены: {out.resolve()}")


if __name__ == "__main__":
    main()
