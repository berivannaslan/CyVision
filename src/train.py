"""Model eğitimi: train/test bölme, SVM / Random Forest / KNN, değerlendirme.

Önce features.csv oluşmuş olmalı (python -m src.features).

    python -m src.train                 # 5. hafta: varsayılan parametrelerle 3 model
    python -m src.train --tune          # 6.-8. hafta: GridSearchCV ile en iyi parametreler
    python -m src.train --tune --models svm   # sadece bir model

Çıktılar (<mod> = varsayilan | ayarli):
    models/<model>_<mod>.joblib            eğitilmiş model (ölçekleyici dahil)
    reports/sonuclar_<mod>.csv             doğruluk, kesinlik, duyarlılık, makro F1, süre
                                           (modeller tek tek eğitilse de hepsi bu dosyada birikir)
    reports/<model>_<mod>_rapor.txt        sınıflandırma raporu (aile bazında)
    reports/aile_f1_<model>_<mod>.csv      aile bazında F1 (karşılaştırma için)
    reports/tahminler_<model>_<mod>.csv    her test örneğinin gerçek ve tahmin edilen ailesi
    reports/cv_<model>.csv                 (--tune) denenen her parametre kombinasyonunun CV sonucu
    reports/figures/05_cm_<model>_<mod>.png karışıklık matrisi
    data/processed/split.csv               hangi örnek train, hangisi test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.features import FEATURES_CSV, load_features  # noqa: E402

REPORTS_DIR = config.ROOT_DIR / "reports"
SPLIT_CSV = config.PROCESSED_DIR / "split.csv"


# ---------------------------------------------------------------- modeller
def build_models() -> dict[str, Pipeline]:
    """Her model bir Pipeline: önce StandardScaler, sonra sınıflandırıcı.

    Ölçekleyici Pipeline içinde olduğu için yalnızca eğitim verisine göre
    ayarlanır; test verisinden bilgi sızmaz (data leakage olmaz).
    """
    rs = config.RANDOM_STATE
    return {
        "svm": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=rs))]),
        "rf": Pipeline([("scaler", StandardScaler()),
                        ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=rs))]),
        "knn": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier())]),
    }


# GridSearchCV için denenecek parametreler (6.-8. hafta)
PARAM_GRIDS = {
    "svm": {"clf__C": [1, 10, 100, 1000], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
    "rf": {"clf__n_estimators": [100, 300, 500], "clf__max_depth": [None, 20, 40],
           "clf__min_samples_leaf": [1, 2]},
    "knn": {"clf__n_neighbors": [1, 3, 5, 7, 9], "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"]},
}

MODEL_NAMES = {"svm": "SVM (RBF)", "rf": "Random Forest", "knn": "KNN"}


# ---------------------------------------------------------------- veri
def split_data(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    """%80 eğitim / %20 test, aile oranları korunarak (stratified).

    Bölme split.csv'ye yazılır; böylece tüm modeller ve sonraki haftalar
    aynı test setini kullanır.
    """
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)
    split = pd.Series("train", index=range(len(y)))
    split.iloc[idx_test] = "test"
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    meta.assign(split=split.values).to_csv(SPLIT_CSV, index=False)
    return (X.iloc[idx_train], X.iloc[idx_test], y.iloc[idx_train], y.iloc[idx_test])


# ---------------------------------------------------------------- değerlendirme
def evaluate(model, X_test, y_test) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test)
    scores = {
        "dogruluk": accuracy_score(y_test, y_pred),
        "kesinlik_makro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "duyarlilik_makro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_makro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_agirlikli": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return scores, y_pred


def save_confusion_matrix(y_test, y_pred, labels, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)  # satır = gerçek aile, oran
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": "Aile içindeki oran"}, annot_kws={"fontsize": 7})
    ax.set_xlabel("Tahmin edilen aile")
    ax.set_ylabel("Gerçek aile")
    ax.set_title(title)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- ana akış
def run(model_keys: list[str], tune: bool, features_csv: Path = FEATURES_CSV,
        n_jobs: int = -1, save: bool = True) -> pd.DataFrame:
    X, y, meta = load_features(features_csv)
    X_train, X_test, y_train, y_test = split_data(X, y, meta)
    labels = sorted(y.unique())
    mode = "ayarli" if tune else "varsayilan"
    print(f"Eğitim: {len(y_train)}  Test: {len(y_test)}  Aile: {len(labels)}  Mod: {mode}\n")

    models = build_models()
    rows = []
    for key in model_keys:
        model = models[key]
        start = time.time()
        best_params = {}
        if tune:
            if key == "rf":
                # GridSearch zaten paralel; içteki RF'yi tek çekirdeğe al (iç içe paralellik yavaşlatır)
                model.set_params(clf__n_jobs=1)
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                                 random_state=config.RANDOM_STATE)
            search = GridSearchCV(model, PARAM_GRIDS[key], cv=cv, scoring="f1_macro",
                                  n_jobs=n_jobs, verbose=1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            print(f"  En iyi parametreler: {best_params}  (CV makro F1 = {search.best_score_:.4f})")
            if save:
                save_cv_results(search, key)
        else:
            model.fit(X_train, y_train)
        train_sec = time.time() - start

        scores, y_pred = evaluate(model, X_test, y_test)
        rows.append({"model": MODEL_NAMES[key], **{k: round(v, 4) for k, v in scores.items()},
                     "egitim_suresi_sn": round(train_sec, 1),
                     "parametreler": json.dumps(best_params, default=str)})
        print(f"{MODEL_NAMES[key]:<14} doğruluk={scores['dogruluk']:.4f}  "
              f"makro F1={scores['f1_makro']:.4f}  ({train_sec:.1f} sn)")

        if save:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, config.MODELS_DIR / f"{key}_{mode}.joblib")
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            (REPORTS_DIR / f"{key}_{mode}_rapor.txt").write_text(
                classification_report(y_test, y_pred, digits=4, zero_division=0), encoding="utf-8")
            save_confusion_matrix(y_test, y_pred, labels,
                                  f"{MODEL_NAMES[key]} ({mode}) - karışıklık matrisi",
                                  config.FIGURES_DIR / f"05_cm_{key}_{mode}.png")
            save_family_scores(y_test, y_pred, key, mode)
            meta.loc[y_test.index, ["path", "family"]].assign(tahmin=y_pred).to_csv(
                REPORTS_DIR / f"tahminler_{key}_{mode}.csv", index=False)

    results = pd.DataFrame(rows)
    if save:
        results = merge_results(results, REPORTS_DIR / f"sonuclar_{mode}.csv")
    print("\n" + results.drop(columns="parametreler").to_string(index=False))
    return results


# ---------------------------------------------------------------- kayıt yardımcıları
def merge_results(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Yeni sonuçları dosyadakilerle birleştirir; aynı modelin eski satırı değiştirilir.

    Böylece modeller tek tek eğitilse de (--models svm, sonra --models rf ...)
    sonuç dosyasında hepsi kalır.
    """
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new["model"])]
        new = pd.concat([old, new], ignore_index=True)
    order = {name: i for i, name in enumerate(MODEL_NAMES.values())}
    new = new.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


def save_cv_results(search: GridSearchCV, key: str) -> Path:
    """GridSearchCV'nin denediği her parametre kombinasyonunun CV sonucunu kaydeder."""
    cv = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in cv.columns if c.startswith("param_")]
    out = cv[param_cols + ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]]
    out = out.rename(columns={c: c.replace("param_clf__", "") for c in param_cols})
    out = out.rename(columns={"mean_test_score": "cv_f1_makro_ort", "std_test_score": "cv_f1_makro_std",
                              "rank_test_score": "sira", "mean_fit_time": "egitim_suresi_sn"})
    path = REPORTS_DIR / f"cv_{key}.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.sort_values("sira").to_csv(path, index=False)
    return path


def save_family_scores(y_test, y_pred, key: str, mode: str) -> Path:
    """Aile bazında kesinlik, duyarlılık, F1 ve örnek sayısını CSV olarak kaydeder."""
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rows = [{"aile": fam, "kesinlik": v["precision"], "duyarlilik": v["recall"],
             "f1": v["f1-score"], "ornek": int(v["support"])}
            for fam, v in rep.items() if isinstance(v, dict) and fam not in ("macro avg", "weighted avg")]
    path = REPORTS_DIR / f"aile_f1_{key}_{mode}.csv"
    pd.DataFrame(rows).round(4).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["svm", "rf", "knn"], choices=["svm", "rf", "knn"])
    ap.add_argument("--tune", action="store_true", help="GridSearchCV ile parametre ayarı")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    run(args.models, args.tune, n_jobs=args.n_jobs)
