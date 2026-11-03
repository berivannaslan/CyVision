"""Varsayılan ve ayarlı (GridSearchCV) modellerin karşılaştırılması.

Önce iki eğitim de yapılmış olmalı:
    python -m src.train            # varsayılan
    python -m src.train --tune     # ayarlı

Sonra:
    python -m src.compare

Çıktılar:
    reports/karsilastirma.csv                     model bazında önce/sonra tablosu
    reports/hatali_ciftler_<model>.csv            en çok karışan aile çiftleri (ayarlı model)
    reports/figures/06_varsayilan_vs_ayarli.png   makro F1 ve doğruluk, önce/sonra
    reports/figures/06_svm_izgara.png             SVM için C × gamma ızgarasında CV makro F1
    reports/figures/06_aile_f1_degisim.png        en iyi modelde aile bazında F1 değişimi
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.train import MODEL_NAMES, REPORTS_DIR  # noqa: E402

KEYS = {v: k for k, v in MODEL_NAMES.items()}  # "Random Forest" -> "rf"


def load_results() -> pd.DataFrame:
    """Varsayılan ve ayarlı sonuç tablolarını yan yana getirir."""
    paths = {m: REPORTS_DIR / f"sonuclar_{m}.csv" for m in ("varsayilan", "ayarli")}
    for m, p in paths.items():
        if not p.exists():
            flag = " --tune" if m == "ayarli" else ""
            raise FileNotFoundError(f"{p} yok. Önce: python -m src.train{flag}")
    v = pd.read_csv(paths["varsayilan"])
    a = pd.read_csv(paths["ayarli"])
    cols = ["dogruluk", "f1_makro"]
    df = v[["model"] + cols].merge(a[["model"] + cols + ["parametreler"]], on="model",
                                   suffixes=("_varsayilan", "_ayarli"))
    for c in cols:
        df[f"{c}_fark"] = (df[f"{c}_ayarli"] - df[f"{c}_varsayilan"]).round(4)
    return df


def plot_before_after(df: pd.DataFrame, path: Path) -> None:
    long = pd.concat([
        df[["model", "f1_makro_varsayilan"]].rename(columns={"f1_makro_varsayilan": "deger"}).assign(ayar="Varsayılan"),
        df[["model", "f1_makro_ayarli"]].rename(columns={"f1_makro_ayarli": "deger"}).assign(ayar="Ayarlı"),
    ])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=long, x="model", y="deger", hue="ayar", ax=ax,
                palette=["#b9b9b9", "#2f6db5"])
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=9, padding=2)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("")
    ax.set_ylabel("Makro F1 (test seti)")
    ax.set_title("Parametre ayarının makro F1'e etkisi")
    ax.legend(title="", loc="upper center", ncol=2, frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_svm_grid(path: Path) -> bool:
    """SVM'nin C × gamma ızgarasında CV makro F1 ısı haritası. cv_svm.csv yoksa atlar."""
    cv_path = REPORTS_DIR / "cv_svm.csv"
    if not cv_path.exists():
        return False
    cv = pd.read_csv(cv_path)
    grid = cv.pivot_table(index="C", columns="gamma", values="cv_f1_makro_ort")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(grid, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                cbar_kws={"label": "CV makro F1 (5 katlı ort.)"})
    ax.set_title("SVM: C ve gamma değerlerine göre çapraz doğrulama sonucu")
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return True


def family_change(key: str) -> pd.DataFrame:
    v = pd.read_csv(REPORTS_DIR / f"aile_f1_{key}_varsayilan.csv")
    a = pd.read_csv(REPORTS_DIR / f"aile_f1_{key}_ayarli.csv")
    df = v[["aile", "f1", "ornek"]].merge(a[["aile", "f1"]], on="aile", suffixes=("_varsayilan", "_ayarli"))
    df["fark"] = (df["f1_ayarli"] - df["f1_varsayilan"]).round(4)
    return df.sort_values("f1_ayarli").reset_index(drop=True)


def plot_family_change(df: pd.DataFrame, model_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    y = range(len(df))
    ax.scatter(df["f1_varsayilan"], y, color="#b9b9b9", label="Varsayılan", zorder=3)
    ax.scatter(df["f1_ayarli"], y, color="#2f6db5", label="Ayarlı", zorder=3)
    for i, r in df.iterrows():
        ax.plot([r["f1_varsayilan"], r["f1_ayarli"]], [i, i], color="#cccccc", zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{a} ({n})" for a, n in zip(df["aile"], df["ornek"])], fontsize=8)
    ax.set_xlabel("F1 (test seti)")
    ax.set_title(f"{model_name}: aile bazında F1, ayar öncesi ve sonrası")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def confused_pairs(key: str, mode: str = "ayarli", top: int = 10) -> pd.DataFrame:
    """En çok karıştırılan (gerçek, tahmin) aile çiftleri."""
    t = pd.read_csv(REPORTS_DIR / f"tahminler_{key}_{mode}.csv")
    wrong = t[t["family"] != t["tahmin"]]
    pairs = (wrong.groupby(["family", "tahmin"]).size().rename("hata").reset_index()
             .sort_values("hata", ascending=False).head(top))
    return pairs.rename(columns={"family": "gercek_aile", "tahmin": "tahmin_edilen"})


def main() -> pd.DataFrame:
    df = load_results()
    df.to_csv(REPORTS_DIR / "karsilastirma.csv", index=False)
    print("Varsayılan ve ayarlı modeller:\n")
    print(df.drop(columns="parametreler").to_string(index=False))
    print("\nEn iyi parametreler:")
    for _, r in df.iterrows():
        print(f"  {r['model']:<14} {r['parametreler']}")

    fig_dir = config.FIGURES_DIR
    plot_before_after(df, fig_dir / "06_varsayilan_vs_ayarli.png")
    if plot_svm_grid(fig_dir / "06_svm_izgara.png"):
        print(f"\nSVM ızgara grafiği -> {fig_dir / '06_svm_izgara.png'}")

    best = df.sort_values("f1_makro_ayarli", ascending=False).iloc[0]
    key = KEYS[best["model"]]
    fam = family_change(key)
    plot_family_change(fam, best["model"], fig_dir / "06_aile_f1_degisim.png")
    print(f"\nEn iyi model (ayarlı): {best['model']}  makro F1 = {best['f1_makro_ayarli']:.4f}")
    print("F1'i en çok değişen aileler:")
    print(fam.reindex(fam["fark"].abs().sort_values(ascending=False).index).head(6).to_string(index=False))

    pairs = confused_pairs(key)
    pairs.to_csv(REPORTS_DIR / f"hatali_ciftler_{key}.csv", index=False)
    print(f"\nEn çok karışan aile çiftleri ({best['model']}, ayarlı):")
    print(pairs.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
