"""Data profiling, training and prediction.

Everything a trained model needs lives in a plain scikit-learn Pipeline, so a
downloaded model file can be loaded anywhere with joblib + scikit-learn alone.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MAX_TRAIN_ROWS = 50_000
MAX_CLASSES = 50
MIN_ROWS = 20
RANDOM_STATE = 42

MODELS = {
    "auto": "Best pick (automatic)",
    "linear": "Simple formula",
    "tree": "Decision tree",
    "forest": "Random forest",
    "boosting": "Gradient boosting",
    "knn": "Nearest neighbours",
}
STEPS = {"dedupe", "drop_missing", "scale", "balance"}


class UserError(ValueError):
    """A problem the person using the app can fix; the message is shown as-is."""


# ---------------------------------------------------------------- profiling


def _is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def guess_task(s: pd.Series) -> str:
    """A number with many different values is a quantity; anything else is a category."""
    return "regression" if _is_number(s) and s.nunique(dropna=True) > 15 else "classification"


def _clean(v: Any) -> Any:
    """Make a value safe for JSON."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    return str(v)


def profile(df: pd.DataFrame) -> dict:
    n = len(df)
    columns = []
    for name in df.columns:
        s = df[name]
        unique = int(s.nunique(dropna=True))
        number = _is_number(s)
        columns.append(
            {
                "name": str(name),
                "type": "number" if number else "text",
                "missing": int(s.isna().sum()),
                "unique": unique,
                "task": guess_task(s),
                # Text that is different on nearly every row (names, IDs) cannot be learned from.
                "id_like": (not number) and n > 0 and unique > max(MAX_CLASSES, 0.5 * n),
                "empty": unique <= 1,
            }
        )
    preview = [[_clean(v) for v in row] for row in df.head(8).itertuples(index=False, name=None)]
    return {"rows": n, "columns": columns, "preview": preview}


# ----------------------------------------------------------------- training


def _label_series(s: pd.Series) -> pd.Series:
    """Class labels as tidy strings (1.0 -> '1')."""
    if _is_number(s) and (s.dropna() % 1 == 0).all():
        s = s.astype("Int64")
    return s.astype(str)


def _prepare_features(df: pd.DataFrame, features: list[dict]) -> pd.DataFrame:
    """Coerce raw data into the dtypes the pipeline was trained on."""
    out = {}
    for f in features:
        col = df[f["name"]] if f["name"] in df.columns else pd.Series([None] * len(df), index=df.index)
        if f["type"] == "number":
            out[f["name"]] = pd.to_numeric(col, errors="coerce").astype(float)
        else:
            text = col.astype(object).where(col.notna(), np.nan)
            out[f["name"]] = text.map(lambda v: v if isinstance(v, float) else str(v)).astype(object)
    return pd.DataFrame(out, index=df.index)


def _estimator(kind: str, task: str, balance: bool):
    cw = "balanced" if balance else None
    if task == "classification":
        return {
            "linear": lambda: LogisticRegression(max_iter=1000, class_weight=cw),
            "tree": lambda: DecisionTreeClassifier(max_depth=6, class_weight=cw, random_state=RANDOM_STATE),
            "forest": lambda: RandomForestClassifier(
                n_estimators=150, class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE
            ),
            "boosting": lambda: HistGradientBoostingClassifier(class_weight=cw, random_state=RANDOM_STATE),
            "knn": lambda: KNeighborsClassifier(n_neighbors=7),
        }[kind]()
    return {
        "linear": lambda: Ridge(),
        "tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "forest": lambda: RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE),
        "boosting": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "knn": lambda: KNeighborsRegressor(n_neighbors=7),
    }[kind]()


def _pipeline(kind: str, task: str, numeric: list[str], text: list[str], scale: bool, balance: bool) -> Pipeline:
    # Distance- and formula-based learners only work sensibly on scaled numbers.
    scale = scale or kind in ("linear", "knn")
    num_steps: list = [("fill", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("numbers", Pipeline(num_steps), numeric))
    if text:
        transformers.append(
            (
                "text",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="constant", fill_value="(blank)")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist", max_categories=25, sparse_output=False
                            ),
                        ),
                    ]
                ),
                text,
            )
        )
    return Pipeline([("prepare", ColumnTransformer(transformers)), ("model", _estimator(kind, task, balance))])


def train(
    df: pd.DataFrame,
    target: str,
    ignore: list[str] | None = None,
    steps: list[str] | None = None,
    model: str = "auto",
    task: str = "auto",
) -> tuple[Pipeline, dict]:
    """Train a model. Returns the fitted pipeline and a JSON-safe description of it."""
    started = time.time()
    ignore, steps = set(ignore or []), set(steps or [])
    notes: list[str] = []

    if target not in df.columns:
        raise UserError("Pick the column you want to predict first.")
    if model not in MODELS:
        raise UserError(f"Unknown learner '{model}'.")
    if steps - STEPS:
        raise UserError(f"Unknown clean-up step: {', '.join(sorted(steps - STEPS))}.")
    if task not in ("auto", "classification", "regression"):
        raise UserError("Unknown kind of prediction.")

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    missing_target = int(df[target].isna().sum())
    if missing_target:
        df = df[df[target].notna()]
        notes.append(f"Skipped {missing_target:,} rows where '{target}' was blank.")
    if "dedupe" in steps:
        before = len(df)
        df = df.drop_duplicates()
        if before - len(df):
            notes.append(f"Removed {before - len(df):,} duplicate rows.")

    info = {c["name"]: c for c in profile(df)["columns"]}
    used, skipped = [], []
    for name in df.columns:
        if name == target or name in ignore:
            continue
        if info[name]["empty"]:
            skipped.append(f"'{name}' (same value on every row)")
        elif info[name]["id_like"]:
            skipped.append(f"'{name}' (different on almost every row, like a name or ID)")
        else:
            used.append(name)
    if skipped:
        notes.append("Left out columns that cannot help: " + "; ".join(skipped) + ".")
    if not used:
        raise UserError("There are no usable columns left to learn from. Un-ignore some columns.")

    if "drop_missing" in steps:
        before = len(df)
        df = df.dropna(subset=used)
        if before - len(df):
            notes.append(f"Removed {before - len(df):,} rows that had blanks.")

    if task == "auto":
        task = guess_task(df[target])
    if task == "regression" and not _is_number(df[target]):
        raise UserError(f"'{target}' contains text, so it can only be predicted as a category.")

    if len(df) < MIN_ROWS:
        raise UserError(f"Only {len(df)} usable rows. At least {MIN_ROWS} are needed to learn anything.")
    if len(df) > MAX_TRAIN_ROWS:
        notes.append(f"Used a random sample of {MAX_TRAIN_ROWS:,} of the {len(df):,} rows to keep training fast.")
        df = df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

    features = [{"name": n, "type": info[n]["type"]} for n in used]
    X = _prepare_features(df, features)
    numeric = [f["name"] for f in features if f["type"] == "number"]
    text = [f["name"] for f in features if f["type"] == "text"]

    stratify = None
    if task == "classification":
        y = _label_series(df[target])
        counts = y.value_counts()
        if len(counts) < 2:
            raise UserError(f"'{target}' has only one value, so there is nothing to predict.")
        if len(counts) > MAX_CLASSES:
            raise UserError(
                f"'{target}' has {len(counts)} different values, which is too many categories to predict. "
                "Pick a column with fewer possible answers."
            )
        if counts.min() >= 2:
            stratify = y
    else:
        y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    scale, balance = "scale" in steps, "balance" in steps

    leaderboard = []
    if model == "auto":
        n = min(len(X_train), 5000)
        Xs, ys = X_train.iloc[:n], y_train.iloc[:n]
        can_cv = task == "regression" or ys.value_counts().min() >= 3
        best, best_score = "forest", -math.inf
        if can_cv:
            scoring = "accuracy" if task == "classification" else "r2"
            for kind in ("linear", "forest", "boosting"):
                try:
                    score = float(
                        cross_val_score(
                            _pipeline(kind, task, numeric, text, scale, balance), Xs, ys, cv=3, scoring=scoring
                        ).mean()
                    )
                except Exception:  # one learner failing should not sink the others
                    continue
                if math.isnan(score):
                    continue
                leaderboard.append({"model": kind, "label": MODELS[kind], "score": round(score, 4)})
                if score > best_score:
                    best, best_score = kind, score
            leaderboard.sort(key=lambda r: -r["score"])
        model = best

    pipe = _pipeline(model, task, numeric, text, scale, balance)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if task == "classification":
        labels = [str(c) for c in pipe.classes_]
        acc = float(accuracy_score(y_test, pred))
        baseline = float((y_test == y_train.value_counts().idxmax()).mean())
        metrics = {
            "accuracy": round(acc, 4),
            "baseline": round(baseline, 4),
            "labels": labels,
            "confusion": confusion_matrix(y_test, pred, labels=pipe.classes_).tolist(),
        }
    else:
        mae = float(mean_absolute_error(y_test, pred))
        baseline = float(mean_absolute_error(y_test, np.full(len(y_test), y_train.median())))
        pick = np.random.default_rng(RANDOM_STATE).permutation(len(y_test))[:200]
        metrics = {
            "r2": round(float(r2_score(y_test, pred)), 4),
            "mae": mae,
            "baseline_mae": baseline,
            "points": [[float(y_test.iloc[i]), float(pred[i])] for i in pick],
        }

    # Which columns matter: shuffle one column at a time and see how much worse the model gets.
    sample = min(len(X_test), 300 if len(used) <= 60 else 120)
    imp = permutation_importance(
        pipe, X_test.iloc[:sample], y_test.iloc[:sample], n_repeats=3, random_state=RANDOM_STATE
    ).importances_mean
    imp = np.clip(imp, 0, None)
    total = float(imp.sum())
    importance = sorted(
        ({"name": n, "share": round(float(v) / total, 4) if total else 0.0} for n, v in zip(used, imp)),
        key=lambda r: -r["share"],
    )

    for f in features:
        col = X[f["name"]]
        if f["type"] == "number":
            f["integer"] = bool((col.dropna() % 1 == 0).all())
            f["default"] = _clean(round(float(col.median()), 0 if f["integer"] else 4))
            f["min"], f["max"] = _clean(col.min()), _clean(col.max())
        else:
            top = col.dropna().value_counts()
            f["choices"] = [str(v) for v in top.index[:25]]
            f["default"] = f["choices"][0] if f["choices"] else ""

    meta = {
        "task": task,
        "target": target,
        "model": model,
        "model_label": MODELS[model],
        "steps": sorted(steps),
        "features": features,
        "rows_trained": len(X_train),
        "rows_tested": len(X_test),
        "metrics": metrics,
        "importance": importance,
        "leaderboard": leaderboard,
        "notes": notes,
        "seconds": round(time.time() - started, 2),
    }
    return pipe, meta


# --------------------------------------------------------------- prediction


def predict(pipe: Pipeline, meta: dict, rows: pd.DataFrame) -> list[dict]:
    if rows.empty:
        return []
    X = _prepare_features(rows, meta["features"])
    pred = pipe.predict(X)
    if meta["task"] == "regression":
        return [{"prediction": _clean(float(v))} for v in pred]
    out = [{"prediction": str(v)} for v in pred]
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        for row, p in zip(out, proba):
            row["confidence"] = round(float(p.max()), 4)
            row["chances"] = {str(c): round(float(v), 4) for c, v in zip(pipe.classes_, p)}
    return out
