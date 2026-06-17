# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

# =========================
# НАСТРОЙКИ
# =========================
DATE_COL = "date"
TARGET_COL = "target"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"
TEST_END = None  
S = 12

ORDERS = [(a, 1, b) for a in [0, 1] for b in [0, 1]]
SEASONAL_ORDERS = [(c, 0, d, 12) for c in [0, 1] for d in [0, 1]]
MAXITER = 1000
OPTIMIZER_METHOD = "lbfgs"
USE_ACTUAL_FUTURE_EXOG = False
PAUSE_AT_END = True

OUTPUT_DIR_NAME = "results_sarimax_64_recursive"
OUTPUT_XLSX_NAME = "sarimax_64_recursive_results.xlsx"

INPUT_CANDIDATES = [
    ("Матрица данных по пшенице.xlsx", None)
  ]

EXOG_SETS: Dict[str, List[str]] = {
    "market_regulation": ["world_price", "usd_rub", "duty"],
    "cost_macro_logistics": ["duty", "diesel", "cpi", "freight_tariff"],
    "full_economic": ["duty", "diesel", "cpi", "freight_tariff", "world_price", "usd_rub"],
    "duty_only": ["duty"],
}
REQUIRED_EXOG_COLS = sorted(set(sum(EXOG_SETS.values(), [])))


def get_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def make_output_dir(base_dir: Path) -> Path:
    out = base_dir / OUTPUT_DIR_NAME
    try:
        out.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out = Path.home() / "Documents" / OUTPUT_DIR_NAME
        out.mkdir(parents=True, exist_ok=True)
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError("В данных должны быть столбцы date и target")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL].astype(str), errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError("Не удалось распознать часть дат в столбце date")
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    df.index.name = "date"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "month" not in df.columns:
        df["month"] = df.index.month
    return df


def load_data(base_dir: Path) -> Tuple[pd.DataFrame, str, str]:
    errors = []
    for filename, sheet_name in INPUT_CANDIDATES:
        path = base_dir / filename
        if not path.exists():
            errors.append(f"{filename}: файл не найден")
            continue
        try:
            if sheet_name is None:
                raw = pd.read_excel(path)
            else:
                raw = pd.read_excel(path, sheet_name=sheet_name)
            df = normalize_columns(raw)
            missing = [c for c in REQUIRED_EXOG_COLS if c not in df.columns]
            if missing:
                errors.append(f"{filename}: нет столбцов {missing}")
                continue
            return df, str(path), sheet_name or "first_sheet"
        except Exception as e:
            errors.append(f"{filename}: {e}")
    raise FileNotFoundError("Не удалось найти данные со всеми нужными столбцами.\n" + "\n".join(errors))


def get_test_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    start = pd.Timestamp(TEST_START)
    if TEST_END is None:
        avail = df[df.index.year == start.year].index
        if len(avail) == 0:
            raise ValueError("В данных нет наблюдений за тестовый год")
        end = avail.max()
    else:
        end = pd.Timestamp(TEST_END)
    return pd.date_range(start, end, freq="MS").intersection(df.index)


def make_future_exog(
    df: pd.DataFrame,
    cols: List[str],
    train_end: pd.Timestamp,
    test_dates: pd.DatetimeIndex,
    use_actual_future_exog: bool = False,
) -> pd.DataFrame:
  
    if use_actual_future_exog:
        return df.loc[test_dates, cols].copy()

    last = df.loc[train_end]
    rows = []
    for d in test_dates:
        rows.append({col: last[col] for col in cols})
    return pd.DataFrame(rows, index=test_dates, columns=cols)


def calc_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)
    mask = y_true.notna() & y_pred.notna() & (y_true != 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "n_test_obs": int(len(y_true)),
    }


def residual_diagnostics(resid, model_id: str, drop_initial: int = 1) -> Tuple[Dict[str, float], pd.DataFrame]:
    r = pd.Series(resid).replace([np.inf, -np.inf], np.nan).dropna()
    if drop_initial > 0 and len(r) > drop_initial + 10:
        r = r.iloc[drop_initial:]

    out: Dict[str, float] = {}
    rows = []

    for lag in [6, 12]:
        try:
            lb = acorr_ljungbox(r, lags=[lag], return_df=True)
            stat = float(lb["lb_stat"].iloc[0])
            pval = float(lb["lb_pvalue"].iloc[0])
            out[f"LB{lag}_stat"] = stat
            out[f"LB{lag}_pvalue"] = pval
            rows.append({
                "model_id": model_id,
                "test": "Ljung-Box",
                "lag": lag,
                "statistic": stat,
                "p_value": pval,
                "conclusion_5pct": "автокорреляция не выявлена" if pval >= 0.05 else "есть признаки автокорреляции",
            })
        except Exception as e:
            rows.append({"model_id": model_id, "test": "Ljung-Box", "lag": lag, "error": str(e)})

    try:
        adf = adfuller(r, autolag="AIC")
        out["ADF_stat"] = float(adf[0])
        out["ADF_pvalue"] = float(adf[1])
        rows.append({
            "model_id": model_id,
            "test": "ADF",
            "lag": "-",
            "statistic": float(adf[0]),
            "p_value": float(adf[1]),
            "conclusion_5pct": "остатки стационарны" if adf[1] < 0.05 else "стационарность не подтверждена",
        })
    except Exception as e:
        rows.append({"model_id": model_id, "test": "ADF", "lag": "-", "error": str(e)})

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kp = kpss(r, regression="c", nlags="auto")
        out["KPSS_stat"] = float(kp[0])
        out["KPSS_pvalue"] = float(kp[1])
        rows.append({
            "model_id": model_id,
            "test": "KPSS",
            "lag": "-",
            "statistic": float(kp[0]),
            "p_value": float(kp[1]),
            "conclusion_5pct": "гипотеза стационарности не отвергается" if kp[1] >= 0.05 else "есть признаки нестационарности",
        })
    except Exception as e:
        rows.append({"model_id": model_id, "test": "KPSS", "lag": "-", "error": str(e)})

    return out, pd.DataFrame(rows)


def fit_one(
    df: pd.DataFrame,
    exog_set_name: str,
    exog_cols: List[str],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    train_end: pd.Timestamp,
    test_dates: pd.DatetimeIndex,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[:train_end, [TARGET_COL] + exog_cols].dropna()
    y_train = train[TARGET_COL]
    if len(y_train) < 20:
        raise ValueError(f"Слишком мало наблюдений после dropna: {len(y_train)}")

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train[exog_cols]), index=train.index, columns=exog_cols)
    X_future_raw = make_future_exog(df, exog_cols, train_end, test_dates, USE_ACTUAL_FUTURE_EXOG)
    X_future = pd.DataFrame(scaler.transform(X_future_raw), index=X_future_raw.index, columns=exog_cols)

    n = seasonal_order[0]
    model_id = f"SARIMAX{order}{seasonal_order} | {exog_set_name}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=MAXITER, method=OPTIMIZER_METHOD)
        warning_text = " | ".join(sorted(set(str(w.message) for w in caught)))[:1500]

    pred = res.get_forecast(steps=len(test_dates), exog=X_future).predicted_mean
    pred.index = test_dates
    actual = df.loc[test_dates, TARGET_COL]
    metrics = calc_metrics(actual, pred)

    diag, diag_long = residual_diagnostics(res.resid, model_id, drop_initial=max(1, order[1]))

    mle = res.mle_retvals if hasattr(res, "mle_retvals") else {}
    row = {
        "model_id": model_id,
        "specification": f"SARIMAX{order}{seasonal_order}",
        "order": str(order),
        "seasonal_order": str(seasonal_order),
        "p": order[0],
        "d": order[1],
        "q": order[2],
        "P": seasonal_order[0],
        "D": seasonal_order[1],
        "Q": seasonal_order[2],
        "S": seasonal_order[3],
        "n_in_seasonal_part": n,
        "exog_set": exog_set_name,
        "exog_cols": ", ".join(exog_cols),
        "n_exog": len(exog_cols),
        "train_start": y_train.index.min(),
        "train_end": y_train.index.max(),
        "train_n": len(y_train),
        "test_start": test_dates.min(),
        "test_end": test_dates.max(),
        "test_n": len(test_dates),
        "future_exog_mode": "actual_2025_exog" if USE_ACTUAL_FUTURE_EXOG else "fixed_at_train_end",
        "optimizer_method": OPTIMIZER_METHOD,
        "maxiter": MAXITER,
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "HQIC": float(res.hqic) if hasattr(res, "hqic") else np.nan,
        "converged": bool(mle.get("converged", False)),
        "iterations": mle.get("iterations", None),
        "function_calls": mle.get("fcalls", None),
        "warnflag": mle.get("warnflag", None),
        "warnings": warning_text,
        **metrics,
        **diag,
    }
    row["LB6_ok_5pct"] = bool(row.get("LB6_pvalue", np.nan) >= 0.05) if pd.notna(row.get("LB6_pvalue", np.nan)) else False
    row["LB12_ok_5pct"] = bool(row.get("LB12_pvalue", np.nan) >= 0.05) if pd.notna(row.get("LB12_pvalue", np.nan)) else False
    row["ADF_ok_5pct"] = bool(row.get("ADF_pvalue", np.nan) < 0.05) if pd.notna(row.get("ADF_pvalue", np.nan)) else False
    row["KPSS_ok_5pct"] = bool(row.get("KPSS_pvalue", np.nan) >= 0.05) if pd.notna(row.get("KPSS_pvalue", np.nan)) else False
    row["diagnostics_pass_5pct"] = bool(row["LB6_ok_5pct"] and row["LB12_ok_5pct"] and row["ADF_ok_5pct"] and row["KPSS_ok_5pct"])

    forecast_rows = []
    for dte, y, yhat in zip(test_dates, actual, pred):
        forecast_rows.append({
            "model_id": model_id,
            "date": dte,
            "actual": float(y),
            "predicted": float(yhat),
            "error_pred_minus_actual": float(yhat - y),
            "abs_error": float(abs(yhat - y)),
            "ape_%": float(abs((y - yhat) / y) * 100) if y != 0 else np.nan,
        })
    forecasts_df = pd.DataFrame(forecast_rows)

    params_df = pd.DataFrame({
        "model_id": model_id,
        "parameter": res.params.index.astype(str),
        "estimate": res.params.values,
    })

    return row, forecasts_df, diag_long, params_df


def make_table_ready(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    out = results.copy().sort_values(
        ["diagnostics_pass_5pct", "converged", "MAPE_%"],
        ascending=[False, False, True],
    )
    cols = [
        "specification", "exog_set", "exog_cols",
        "MAE", "RMSE", "MAPE_%", "R2", "AIC", "BIC",
        "converged", "iterations", "function_calls", "warnflag",
        "LB6_pvalue", "LB12_pvalue", "ADF_pvalue", "KPSS_pvalue", "diagnostics_pass_5pct",
        "future_exog_mode", "warnings",
    ]
    return out[[c for c in cols if c in out.columns]]


def save_excel(path: Path, readme, results, forecasts, diagnostics, params, errors, prepared):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        make_table_ready(results).to_excel(writer, sheet_name="Table_3_1_ready", index=False)
        results.sort_values("MAPE_%", na_position="last").to_excel(writer, sheet_name="SARIMAX_checked", index=False)
        forecasts.to_excel(writer, sheet_name="Forecasts_long", index=False)
        diagnostics.to_excel(writer, sheet_name="Residual_diagnostics", index=False)
        params.to_excel(writer, sheet_name="Estimated_parameters", index=False)
        errors.to_excel(writer, sheet_name="Errors", index=False)
        prepared.reset_index().to_excel(writer, sheet_name="Prepared_data", index=False)

    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    border = Border(
        left=Side(style="thin", color="DDDDDD"),
        right=Side(style="thin", color="DDDDDD"),
        top=Side(style="thin", color="DDDDDD"),
        bottom=Side(style="thin", color="DDDDDD"),
    )
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        if ws.max_row >= 1 and ws.max_column >= 1:
            ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
            try:
                safe_title = "".join(ch if ch.isalnum() else "_" for ch in ws.title)[:20]
                tab = Table(displayName=f"T_{safe_title}", ref=ref)
                tab.tableStyleInfo = TableStyleInfo(
                    name="TableStyleMedium2",
                    showFirstColumn=False,
                    showLastColumn=False,
                    showRowStripes=True,
                    showColumnStripes=False,
                )
                ws.add_table(tab)
            except Exception:
                pass
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows():
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col_cells in ws.columns:
            col_letter = get_column_letter(col_cells[0].column)
            max_len = 0
            for cell in col_cells[:100]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 60)
    wb.save(path)


def main():
    base = get_base_dir()
    out_dir = make_output_dir(base)
    df, source_path, source_sheet = load_data(base)
    train_end = pd.Timestamp(TRAIN_END)
    test_dates = get_test_dates(df)

    rows = []
    forecasts = []
    diagnostics = []
    params = []
    errors = []

    print(f"Источник данных: {source_path} | sheet={source_sheet}")
    print(f"Train до: {train_end.date()}")
    print(f"Test: {test_dates.min().date()} — {test_dates.max().date()}, n={len(test_dates)}")
    print(f"orders={ORDERS}; seasonal_orders={SEASONAL_ORDERS}; maxiter={MAXITER}; future_exog={('actual' if USE_ACTUAL_FUTURE_EXOG else 'fixed_at_train_end')}")

    candidates = []
    for order in ORDERS:
        for seasonal_order in SEASONAL_ORDERS:
            for exog_set_name, exog_cols in EXOG_SETS.items():
                candidates.append((order, seasonal_order, exog_set_name, exog_cols))

    for i, (order, seasonal_order, exog_set_name, exog_cols) in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] SARIMAX{order}{seasonal_order} | {exog_set_name}: {exog_cols}")
        try:
            row, fdf, ddf, pdf = fit_one(df, exog_set_name, exog_cols, order, seasonal_order, train_end, test_dates)
            rows.append(row)
            forecasts.append(fdf)
            diagnostics.append(ddf)
            params.append(pdf)
            print(f"   MAPE={row['MAPE_%']:.3f}%, R2={row['R2']:.3f}, AIC={row['AIC']:.2f}, BIC={row['BIC']:.2f}, converged={row['converged']}, LB12_p={row.get('LB12_pvalue', np.nan):.4f}, pass={row['diagnostics_pass_5pct']}")
        except Exception as e:
            errors.append({
                "specification": f"SARIMAX{order}{seasonal_order}",
                "exog_set": exog_set_name,
                "exog_cols": ", ".join(exog_cols),
                "error": str(e),
            })
            print(f"   ОШИБКА: {e}")

    results_df = pd.DataFrame(rows)
    forecasts_df = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    diagnostics_df = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    params_df = pd.concat(params, ignore_index=True) if params else pd.DataFrame()
    errors_df = pd.DataFrame(errors)

    readme = pd.DataFrame([
        {"parameter": "purpose", "value": "Проверка SARIMAX(a,1,b)(c,0,d,12), a,b,c,d=0 или 1, для 4 наборов exog; всего 64 вариации"},
        {"parameter": "source_path", "value": source_path},
        {"parameter": "source_sheet", "value": source_sheet},
        {"parameter": "orders", "value": str(ORDERS)},
        {"parameter": "seasonal_orders", "value": str(SEASONAL_ORDERS)},
        {"parameter": "maxiter", "value": MAXITER},
        {"parameter": "optimizer", "value": OPTIMIZER_METHOD},
        {"parameter": "TRAIN_END", "value": TRAIN_END},
        {"parameter": "TEST_START", "value": TEST_START},
        {"parameter": "TEST_END", "value": str(TEST_END)},
        {"parameter": "future_exog_mode", "value": "actual_2025_exog" if USE_ACTUAL_FUTURE_EXOG else "fixed_at_train_end"},
        {"parameter": "diagnostics_pass_5pct", "value": "True если LB6 p>=0.05, LB12 p>=0.05, ADF p<0.05, KPSS p>=0.05"},
    ])

    out_xlsx = out_dir / OUTPUT_XLSX_NAME
    save_excel(out_xlsx, readme, results_df, forecasts_df, diagnostics_df, params_df, errors_df, df)

    print("\nГотово.")
    print(f"Excel: {out_xlsx}")
    if not results_df.empty:
        print("\nТаблица лидеров:")
        print(make_table_ready(results_df).head(20).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nОШИБКА:")
        print(e)
        raise
    finally:
        if PAUSE_AT_END:
            try:
                input("\nНажмите Enter, чтобы закрыть окно...")
            except (EOFError, OSError):
                pass
