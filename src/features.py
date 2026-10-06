"""Doku özelliklerinin çıkarımı: LBP (59) + GLCM (24) = 83 boyutlu vektör.

Tüm veri seti için (önce python -m src.preprocess çalıştırılmış olmalı):
    python -m src.features            # -> data/processed/features.csv
    python -m src.features --n-jobs 4 # paralel (daha hızlı)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocess import PROCESSED_IMG_DIR, read_gray  # noqa: E402

FEATURES_CSV = config.PROCESSED_DIR / "features.csv"

# nri_uniform: P*(P-1) + 3 farklı örüntü -> P=8 için 59
N_LBP_BINS = config.LBP_P * (config.LBP_P - 1) + 3
GLCM_ANGLES_RAD = [np.deg2rad(a) for a in config.GLCM_ANGLES_DEG]


def lbp_feature_names() -> list[str]:
    return [f"lbp_{i}" for i in range(N_LBP_BINS)]


def glcm_feature_names() -> list[str]:
    return [f"glcm_{prop}_{d}_{a}"
            for prop in config.GLCM_PROPS
            for d in config.GLCM_DISTANCES
            for a in config.GLCM_ANGLES_DEG]


def feature_names() -> list[str]:
    return lbp_feature_names() + glcm_feature_names()


def lbp_image(img: np.ndarray) -> np.ndarray:
    """Her piksel için LBP kodunu içeren görüntü (0..58)."""
    return local_binary_pattern(img, P=config.LBP_P, R=config.LBP_R, method=config.LBP_METHOD)


def lbp_features(img: np.ndarray) -> np.ndarray:
    """LBP histogramı: 59 boyut, toplamı 1 olacak şekilde normalize."""
    codes = lbp_image(img)
    hist, _ = np.histogram(codes, bins=N_LBP_BINS, range=(0, N_LBP_BINS))
    return hist.astype(np.float64) / max(hist.sum(), 1)


def glcm_matrix(img: np.ndarray) -> np.ndarray:
    """Simetrik ve normalize GLCM. Boyut: (256, 256, mesafe, açı)."""
    return graycomatrix(img, distances=config.GLCM_DISTANCES, angles=GLCM_ANGLES_RAD,
                        levels=256, symmetric=True, normed=True)


def glcm_features(img: np.ndarray) -> np.ndarray:
    """6 özellik x 4 açı = 24 boyut. Sıra glcm_feature_names() ile aynıdır."""
    m = glcm_matrix(img)
    return np.concatenate([graycoprops(m, prop).ravel() for prop in config.GLCM_PROPS])


def extract_features(img: np.ndarray) -> np.ndarray:
    """Tek görüntü için 83 boyutlu vektör (LBP + GLCM)."""
    if img.dtype != np.uint8:
        raise ValueError("Görüntü uint8 (0-255) olmalı.")
    return np.concatenate([lbp_features(img), glcm_features(img)])


def _extract_from_path(path: str) -> np.ndarray:
    return extract_features(read_gray(path))


def extract_dataset(index_csv: Path = PROCESSED_IMG_DIR / "index.csv",
                    out_csv: Path = FEATURES_CSV, n_jobs: int = 1) -> pd.DataFrame:
    """index.csv'deki tüm görüntülerin özelliklerini çıkarıp CSV'ye yazar.

    Çıktı sütunları: path, family, label_id, lbp_0..lbp_58, glcm_...(24)
    """
    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"{index_csv} yok. Önce: python -m src.preprocess")
    idx = pd.read_csv(index_csv)

    start = time.time()
    if n_jobs == 1:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **_):
                return x
        feats = [_extract_from_path(p) for p in tqdm(idx["path"], desc="Özellik çıkarımı")]
    else:
        from joblib import Parallel, delayed
        feats = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_extract_from_path)(p) for p in idx["path"])

    X = pd.DataFrame(np.vstack(feats), columns=feature_names())
    out = pd.concat([idx.reset_index(drop=True), X], axis=1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"{len(out)} görüntü, {X.shape[1]} özellik, {time.time() - start:.1f} sn -> {out_csv}")
    return out


def load_features(path: Path = FEATURES_CSV) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """features.csv'yi (X, y, meta) olarak döndürür. 5. haftada model eğitiminde kullanılır."""
    df = pd.read_csv(path)
    names = feature_names()
    return df[names], df["family"], df[["path", "family", "label_id"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=1, help="Paralel işlem sayısı (-1: tüm çekirdekler)")
    extract_dataset(n_jobs=ap.parse_args().n_jobs)
