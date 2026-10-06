"""Görüntü ön işleme: gri tonlamalı okuma ve 256x256'ya yeniden boyutlandırma.

Tüm veri setini işleyip data/processed/images_256/<aile>/ altına kaydetmek için:
    python -m src.preprocess
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.data_loader import load_dataset_index  # noqa: E402

PROCESSED_IMG_DIR = config.PROCESSED_DIR / "images_256"


def read_gray(path: str | Path) -> np.ndarray:
    """Görüntüyü gri tonlamalı okur. Windows'ta Türkçe karakterli yollarda da çalışır."""
    buf = np.fromfile(str(path), dtype=np.uint8)  # cv2.imread Unicode yollarda sorun çıkarabilir
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return img


def resize(img: np.ndarray, size: tuple[int, int] = config.IMG_SIZE) -> np.ndarray:
    """Görüntüyü hedef boyuta getirir.

    Küçültmede INTER_AREA (doku bilgisini en iyi korur), büyütmede INTER_LINEAR kullanılır.
    """
    h, w = img.shape[:2]
    shrinking = w > size[0] or h > size[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def preprocess_image(path: str | Path, size: tuple[int, int] = config.IMG_SIZE) -> np.ndarray:
    """Oku + yeniden boyutlandır. Çıktı: (256, 256) uint8."""
    return resize(read_gray(path), size)


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Kaydedilemedi: {path}")
    buf.tofile(str(path))


def preprocess_dataset(df: pd.DataFrame | None = None,
                       out_dir: Path = PROCESSED_IMG_DIR,
                       overwrite: bool = False) -> pd.DataFrame:
    """Tüm veri setini işler ve kaydeder. Daha önce işlenmiş dosyaları atlar.

    Dönüş: path (yeni yol), family, label_id sütunlu tablo; ayrıca
    out_dir/index.csv olarak kaydedilir.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # tqdm yoksa düz döngü
        def tqdm(x, **_):
            return x

    df = load_dataset_index() if df is None else df
    out_dir = Path(out_dir)
    rows, errors = [], []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Ön işleme"):
        dst = out_dir / r["family"] / (Path(r["path"]).stem + ".png")
        if overwrite or not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                write_png(dst, preprocess_image(r["path"]))
            except ValueError as e:
                errors.append(str(e))
                continue
        rows.append({"path": str(dst), "family": r["family"], "label_id": r["label_id"]})

    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "index.csv", index=False)
    if errors:
        print(f"{len(errors)} görüntü okunamadı, ilk hata: {errors[0]}")
    return out


if __name__ == "__main__":
    result = preprocess_dataset()
    print(f"{len(result)} görüntü işlendi -> {PROCESSED_IMG_DIR}")
