from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ELS ratio using enriched inner-ear geometry features.")
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Folder containing ear_geometry_features.csv and volumes_mm3.csv.",
    )
    parser.add_argument("--feature-csv", default=None, help="Override path to ear_geometry_features.csv.")
    parser.add_argument("--volume-csv", default=None, help="Override path to volumes_mm3.csv.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: <analysis-dir>/geometry_model")
    parser.add_argument(
        "--target",
        choices=("els_over_tv", "els_over_tv_plus_els", "els_mm3"),
        default="els_over_tv",
        help="Regression target.",
    )
    parser.add_argument("--max-features", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--predict-feature-csv", default=None, help="Inference feature CSV path.")
    parser.add_argument("--model-path", default=None, help="Saved model bundle (.pkl) for inference.")
    return parser.parse_args()


def infer_analysis_dir(user_value: str | None) -> Path:
    if user_value:
        path = Path(user_value)
        if not path.exists():
            raise FileNotFoundError(f"Analysis directory not found: {path}")
        return path
    for child in Path.cwd().iterdir():
        if not child.is_dir():
            continue
        for candidate in ("analysis_out_geometry", "analysis_out_s20", "analysis_out"):
            analysis_dir = child / candidate
            if (analysis_dir / "ear_geometry_features.csv").exists() and (analysis_dir / "volumes_mm3.csv").exists():
                return analysis_dir
    raise FileNotFoundError("Could not auto-detect an analysis directory with ear_geometry_features.csv.")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    analysis_dir = infer_analysis_dir(args.analysis_dir)
    feature_csv = Path(args.feature_csv) if args.feature_csv else analysis_dir / "ear_geometry_features.csv"
    volume_csv = Path(args.volume_csv) if args.volume_csv else analysis_dir / "volumes_mm3.csv"
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "geometry_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    return feature_csv, volume_csv, output_dir


def build_target_table(volume_csv: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(volume_csv)
    df["pid"] = df["pid"].astype(str).str.zfill(3)
    df["side"] = df["side"].astype(str).str.upper()
    wide = (
        df.pivot_table(index=["pid", "side"], columns="struct", values="volume_mm3", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
    )
    if "ELS" not in wide.columns:
        raise ValueError(f"ELS column is missing from {volume_csv}")
    if target == "els_mm3":
        wide["target_value"] = wide["ELS"].astype(float)
    elif target == "els_over_tv":
        if "TV" not in wide.columns:
            raise ValueError(f"TV column is missing from {volume_csv}")
        wide["target_value"] = wide["ELS"].astype(float) / wide["TV"].astype(float)
    elif target == "els_over_tv_plus_els":
        if "TV" not in wide.columns:
            raise ValueError(f"TV column is missing from {volume_csv}")
        denom = wide["ELS"].astype(float) + wide["TV"].astype(float)
        wide["target_value"] = wide["ELS"].astype(float) / denom
    else:
        raise ValueError(f"Unsupported target: {target}")
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_value"]).copy()
    return wide[["pid", "side", "target_value"]]


def build_feature_table(feature_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(feature_csv)
    df["pid"] = df["pid"].astype(str).str.zfill(3)
    df["side"] = df["side"].astype(str).str.upper()
    if "complete_three_canals" in df.columns:
        df = df[df["complete_three_canals"] == 1].copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_numeric = [col for col in numeric_cols if col != "complete_three_canals"]

    contra = df[["pid", "side", *base_numeric]].copy()
    contra["side"] = contra["side"].map({"L": "R", "R": "L"})
    contra = contra.rename(columns={col: f"contra_{col}" for col in base_numeric})
    merged = df.merge(contra, on=["pid", "side"], how="left")
    for col in base_numeric:
        merged[f"absdiff_{col}"] = (merged[col] - merged[f"contra_{col}"]).abs()
    merged["side_is_right"] = (merged["side"] == "R").astype(int)
    return merged


def build_training_table(feature_csv: Path, volume_csv: Path, target: str) -> tuple[pd.DataFrame, list[str]]:
    feat_df = build_feature_table(feature_csv)
    target_df = build_target_table(volume_csv, target)
    merged = feat_df.merge(target_df, on=["pid", "side"], how="inner")
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != "target_value"]
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged, feature_cols


def make_cv(n_samples: int, random_state: int):
    if n_samples <= 80:
        return LeaveOneOut(), "leave_one_out"
    return KFold(n_splits=5, shuffle=True, random_state=random_state), "kfold_5"


def make_models(feature_count: int, max_features: int, random_state: int) -> dict[str, object]:
    k = max(1, min(max_features, feature_count))
    common_prefix = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
    ]
    models = {
        "ridge": Pipeline(
            common_prefix
            + [
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_regression, k=k)),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        ),
        "elasticnet": Pipeline(
            common_prefix
            + [
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_regression, k=k)),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                        alphas=np.logspace(-3, 1, 25),
                        max_iter=20000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            common_prefix
            + [
                ("select", SelectKBest(score_func=f_regression, k=k)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gb": Pipeline(
            common_prefix
            + [
                ("select", SelectKBest(score_func=f_regression, k=k)),
                ("model", HistGradientBoostingRegressor(random_state=random_state)),
            ]
        ),
    }
    return models


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    y_true_c = y_true - np.mean(y_true)
    y_pred_c = y_pred - np.mean(y_pred)
    denom = np.linalg.norm(y_true_c) * np.linalg.norm(y_pred_c)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(y_true_c, y_pred_c) / denom)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": pearson_r(y_true, y_pred),
    }


def evaluate_models(train_df: pd.DataFrame, feature_cols: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = train_df[feature_cols]
    y = train_df["target_value"].to_numpy(dtype=float)
    cv, cv_name = make_cv(len(train_df), args.random_state)

    metric_rows = []
    prediction_frames = []

    baseline = np.repeat(np.mean(y), len(y))
    metric_rows.append({"model": "baseline_mean", "cv": cv_name, "n": len(train_df), **score_predictions(y, baseline)})
    prediction_frames.append(
        pd.DataFrame({"pid": train_df["pid"], "side": train_df["side"], "model": "baseline_mean", "y_true": y, "y_pred": baseline})
    )

    for model_name, model in make_models(len(feature_cols), args.max_features, args.random_state).items():
        preds = cross_val_predict(model, x, y, cv=cv, n_jobs=1)
        metric_rows.append({"model": model_name, "cv": cv_name, "n": len(train_df), **score_predictions(y, preds)})
        prediction_frames.append(
            pd.DataFrame({"pid": train_df["pid"], "side": train_df["side"], "model": model_name, "y_true": y, "y_pred": preds})
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(by=["rmse", "mae", "r2"], ascending=[True, True, False])
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    return metrics_df, pred_df


def extract_feature_importance(fitted_model, feature_cols: list[str]) -> pd.DataFrame:
    if not hasattr(fitted_model, "named_steps"):
        return pd.DataFrame(columns=["feature", "metric", "value", "abs_value"])
    selected_cols = feature_cols
    if "select" in fitted_model.named_steps:
        selector = fitted_model.named_steps["select"]
        support = selector.get_support()
        selected_cols = [col for col, keep in zip(feature_cols, support) if keep]

    model = fitted_model.named_steps["model"]
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float).reshape(-1)
        metric_name = "coefficient"
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
        metric_name = "importance"
    else:
        return pd.DataFrame(columns=["feature", "metric", "value", "abs_value"])

    out = pd.DataFrame({"feature": selected_cols, "metric": metric_name, "value": values})
    out["abs_value"] = out["value"].abs()
    return out.sort_values("abs_value", ascending=False)


def fit_best_model(train_df: pd.DataFrame, feature_cols: list[str], best_model_name: str, args: argparse.Namespace):
    if best_model_name == "baseline_mean":
        model = DummyRegressor(strategy="mean")
        model.fit(train_df[feature_cols], train_df["target_value"].to_numpy(dtype=float))
        return best_model_name, model
    model = make_models(len(feature_cols), args.max_features, args.random_state)[best_model_name]
    model.fit(train_df[feature_cols], train_df["target_value"].to_numpy(dtype=float))
    return best_model_name, model


def save_model_bundle(path: Path, model_name: str, model, feature_cols: list[str], metadata: dict) -> None:
    bundle = {
        "model_name": model_name,
        "model": model,
        "feature_cols": feature_cols,
        "metadata": metadata,
    }
    with path.open("wb") as f:
        pickle.dump(bundle, f)


def run_training(args: argparse.Namespace) -> None:
    feature_csv, volume_csv, output_dir = resolve_paths(args)
    train_df, feature_cols = build_training_table(feature_csv, volume_csv, args.target)
    metrics_df, pred_df = evaluate_models(train_df, feature_cols, args)
    best_cv_name = str(metrics_df.iloc[0]["model"])
    best_name, fitted_model = fit_best_model(train_df, feature_cols, best_cv_name, args)
    importance_df = extract_feature_importance(fitted_model, feature_cols)

    model_pkl = output_dir / "best_model.pkl"
    train_df.to_csv(output_dir / "training_dataset.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_dir / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "feature_csv": str(feature_csv),
        "volume_csv": str(volume_csv),
        "output_dir": str(output_dir),
        "target": args.target,
        "n_ears": int(len(train_df)),
        "n_features": int(len(feature_cols)),
        "cv_best_model": best_cv_name,
        "best_model": best_name,
        "best_metrics": metrics_df[metrics_df["model"] == best_cv_name].iloc[0].to_dict(),
        "files": {
            "training_dataset": str(output_dir / "training_dataset.csv"),
            "model_comparison": str(output_dir / "model_comparison.csv"),
            "cv_predictions": str(output_dir / "cv_predictions.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "model_bundle": str(model_pkl),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_model_bundle(model_pkl, best_name, fitted_model, feature_cols, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_model_bundle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def run_inference(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise ValueError("--model-path is required with --predict-feature-csv")
    bundle = load_model_bundle(Path(args.model_path))
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    feat_df = build_feature_table(Path(args.predict_feature_csv))
    out = feat_df[["pid", "side"]].copy()
    out["prediction"] = model.predict(feat_df[feature_cols])
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.predict_feature_csv).resolve().parent / "geometry_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "predictions.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(json.dumps({"output_csv": str(out_csv), "n_predictions": int(len(out))}, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.predict_feature_csv:
        run_inference(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
