"""End-to-end training script:  python -m src.train

raw CSV -> validate -> clean -> (save processed) -> train/test split -> compare 3 models
-> select -> evaluate -> save fitted pipeline + metrics + figures.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

from src import eda, evaluate
from src.config import (
    FIGURES_DIR, METRICS_DIR, MODEL_INPUT_COLUMNS, MODEL_PATH, MODEL_VERSION,
    PROCESSED_DATA_PATH, RANDOM_STATE, RAW_DATA_PATH, TARGET, TEST_SIZE,
)
from src.data import clean, dataset_summary, load_raw
from src.preprocessing import build_pipeline

# Ordered simplest -> most complex. Used as a tie-break in model selection.
AUC_TOLERANCE = 0.001


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE),
    }


def select_model(cv_auc: dict) -> str:
    """Pick the model with the best cross-validated ROC-AUC on the TRAINING data only.
    If several are within AUC_TOLERANCE of the best, prefer the simplest (dict order)."""
    best = max(cv_auc.values())
    return next(name for name in cv_auc if cv_auc[name] >= best - AUC_TOLERANCE)


def _dump(obj, path):
    path.write_text(json.dumps(obj, indent=2))


def run(raw_path=RAW_DATA_PATH, model_path=MODEL_PATH) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("1/6 Loading, validating and cleaning data...")
    raw = load_raw(raw_path)
    df, report = clean(raw)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    _dump(report, METRICS_DIR / "cleaning_report.json")
    _dump(dataset_summary(raw, df, report), METRICS_DIR / "dataset_summary.json")
    print(f"    {report['rows_raw']:,} raw rows -> {report['rows_clean']:,} clean rows")

    print("2/6 Saving EDA figures...")
    eda.save_all(df, FIGURES_DIR)

    print("3/6 Splitting BEFORE fitting any transformer (stratified 80/20)...")
    X, y = df[MODEL_INPUT_COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    print("4/6 Cross-validating and fitting candidate models...")
    results, curves, fitted, cv_auc, cv_std = [], {}, {}, {}, {}
    for name, clf in get_models().items():
        pipe = build_pipeline(clf)
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        cv_auc[name], cv_std[name] = float(scores.mean()), float(scores.std())
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        m = evaluate.compute_metrics(y_test, pred, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr, m["roc_auc"])
        fitted[name] = (pipe, pred)
        results.append({"model": name, **m, "cv_roc_auc_mean": cv_auc[name], "cv_roc_auc_std": cv_std[name]})
        print(f"    {name:20s} test AUC {m['roc_auc']:.4f} | recall {m['recall']:.4f} | CV AUC {cv_auc[name]:.4f}")

    comparison = pd.DataFrame(results)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    print("5/6 Selecting and evaluating the final model...")
    selected = select_model(cv_auc)
    pipe, pred = fitted[selected]
    sel_metrics = next(r for r in results if r["model"] == selected)
    cm = evaluate.confusion_dict(y_test, pred)
    evaluate.plot_confusion_matrix(cm, selected, FIGURES_DIR / "confusion_matrix.png")
    evaluate.plot_roc_curves(curves, selected, FIGURES_DIR / "roc_curve.png")
    imp = evaluate.importance_table(pipe, X_test, y_test, RANDOM_STATE)
    imp.to_csv(METRICS_DIR / "feature_importance.csv", index=False)
    evaluate.plot_importance(imp, selected, FIGURES_DIR / "feature_importance.png")
    plt.close("all")

    rationale = (
        f"{selected} was selected using 5-fold cross-validated ROC-AUC on the training split only "
        f"({cv_auc[selected]:.3f} ± {cv_std[selected]:.3f}); the test set was kept for final reporting. "
        f"ROC-AUC measures ranking quality independently of any threshold, unlike accuracy. "
        f"Where models were within {AUC_TOLERANCE} AUC, the simpler model was preferred. "
        f"At the 0.5 threshold it reaches recall {sel_metrics['recall']:.3f} and precision "
        f"{sel_metrics['precision']:.3f}, with {cm['fn']:,} false negatives and {cm['fp']:,} false positives "
        f"out of {len(y_test):,} test records."
    )
    _dump({
        "model_name": selected, "model_version": MODEL_VERSION, "metrics": sel_metrics,
        "confusion_matrix": cm, "test_size": int(len(y_test)), "train_size": int(len(y_train)),
        "selection_rationale": rationale,
    }, METRICS_DIR / "selected_model.json")

    print("6/6 Saving the fitted pipeline...")
    joblib.dump(pipe, model_path)
    print(f"    Saved {selected} pipeline to {model_path}")
    return {"selected": selected, "metrics": sel_metrics}


if __name__ == "__main__":
    run()
