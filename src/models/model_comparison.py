"""
Model comparison — Logistic Regression vs Random Forest vs XGBoost.
Run: python -m src.models.model_comparison
Saves comparison chart to reports/figures/model_comparison.png
"""
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, classification_report,
)
from xgboost import XGBClassifier

from src.config import (
    DATA_RAW_PATH, FIGURES_DIR, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, XGB_PARAMS,
)
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.build_features import build_features

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int), cat_cols),
        ("num", StandardScaler(), num_cols),
    ], remainder="drop")


def run_comparison():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading + preprocessing …")
    raw   = load_data(str(DATA_RAW_PATH))
    ml_df = build_features(clean_data(raw.copy()))
    ml_df[TARGET_COL] = (
        ml_df[TARGET_COL].map({"Yes": 1, "No": 0})
        .fillna(ml_df[TARGET_COL])
    )
    ml_df[TARGET_COL] = pd.to_numeric(ml_df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    X = ml_df.drop(columns=[TARGET_COL])
    y = ml_df[TARGET_COL]

    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(0)
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown").replace("","Unknown")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
    spw = neg / pos

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": Pipeline([
            ("pre", build_preprocessor(cat_cols, num_cols)),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE, C=0.1,
            )),
        ]),
        "Random Forest": Pipeline([
            ("pre", build_preprocessor(cat_cols, num_cols)),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
                max_depth=8, min_samples_leaf=5,
            )),
        ]),
        "XGBoost (tuned)": Pipeline([
            ("pre", build_preprocessor(cat_cols, num_cols)),
            ("clf", XGBClassifier(**{
                **{k: v for k, v in XGB_PARAMS.items()},
                "scale_pos_weight": spw,
            })),
        ]),
    }

    COLORS = {
        "Logistic Regression": "#E24B4A",
        "Random Forest":       "#EF9F27",
        "XGBoost (tuned)":     "#185FA5",
    }

    results = {}
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, pipe in models.items():
        logger.info("Evaluating %s …", name)
        cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fpr, tpr, _    = roc_curve(y_test, y_prob)
        prec, rec, _   = precision_recall_curve(y_test, y_prob)
        test_auc       = roc_auc_score(y_test, y_prob)
        f1             = f1_score(y_test, y_pred)
        precision      = precision_score(y_test, y_pred, zero_division=0)
        recall         = recall_score(y_test, y_pred, zero_division=0)

        results[name] = dict(
            cv_auc_mean=cv_auc.mean(), cv_auc_std=cv_auc.std(),
            test_auc=test_auc, f1=f1, precision=precision, recall=recall,
            fpr=fpr, tpr=tpr, prec_curve=prec, rec_curve=rec,
        )
        logger.info("  CV AUC: %.4f ± %.4f | Test AUC: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f",
                    cv_auc.mean(), cv_auc.std(), test_auc, f1, precision, recall)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#F8FAFC")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr  = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_tbl = fig.add_subplot(gs[1, :])

    # ROC curves
    ax_roc.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5)
    for name, r in results.items():
        ax_roc.plot(r["fpr"], r["tpr"], lw=2, color=COLORS[name],
                    label=f"{name}  AUC={r['test_auc']:.4f}")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve", fontweight="bold")
    ax_roc.legend(fontsize=8); ax_roc.grid(alpha=0.3)
    
    # PR curves
    for name, r in results.items():
        ax_pr.plot(r["rec_curve"], r["prec_curve"], lw=2, color=COLORS[name], label=name)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve", fontweight="bold")
    ax_pr.legend(fontsize=8); ax_pr.grid(alpha=0.3)

    # Bar — CV AUC comparison
    names  = list(results.keys())
    means  = [results[n]["cv_auc_mean"] for n in names]
    stds   = [results[n]["cv_auc_std"]  for n in names]
    colors = [COLORS[n] for n in names]
    bars   = ax_bar.bar(range(len(names)), means, yerr=stds,
                        color=colors, capsize=5, alpha=0.85)
    ax_bar.set_xticks(range(len(names)))
    ax_bar.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
    ax_bar.set_ylabel("CV ROC-AUC")
    ax_bar.set_ylim(min(means) - 0.04, max(means) + 0.04)
    ax_bar.set_title(f"{CV_FOLDS}-Fold CV AUC Comparison", fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Summary table
    ax_tbl.axis("off")
    tbl_data = []
    headers  = ["Model","CV AUC (mean±std)","Test AUC","F1","Precision","Recall","Winner?"]
    best_auc = max(r["test_auc"] for r in results.values())
    for name, r in results.items():
        # from:
        is_best = "🏆" if r["test_auc"] == best_auc else ""
        # to:
        is_best = "BEST" if r["test_auc"] == best_auc else ""
        tbl_data.append([
            name,
            f"{r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f}",
            f"{r['test_auc']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            is_best,
        ])
    tbl = ax_tbl.table(cellText=tbl_data, colLabels=headers,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1E3A5F"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EFF6FF")
        cell.set_edgecolor("#CBD5E1")

    fig.suptitle("ChurnGuard AI — Model Comparison", fontsize=15,
                 fontweight="bold", y=0.98, color="#1E3A5F")

    out = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Comparison chart saved → %s", out)

    # ── Save results JSON ─────────────────────────────────────────────────────
    summary = {
        name: {k: round(float(v), 4) for k, v in r.items()
               if k not in ("fpr","tpr","prec_curve","rec_curve")}
        for name, r in results.items()
    }
    out_json = FIGURES_DIR.parent / "model_comparison.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info("Results JSON → %s", out_json)

    return results


if __name__ == "__main__":
    run_comparison()
