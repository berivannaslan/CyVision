"""Malimg veri setini okuyan yardımcı fonksiyonlar.

Beklenen yapı (her aile kendi klasöründe):
    data/raw/malimg_paths/
        Adialer.C/      -> *.png
        Agent.FYI/      -> *.png
        ...
Klasör bir seviye daha içte olsa da (ör. malimg_paths/malimg_paths/...) sorun olmaz;
etiket her zaman görüntünün bulunduğu klasörün adıdır.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_dataset_index(root: Path | str = config.MALIMG_DIR) -> pd.DataFrame:
    """Tüm görüntülerin yolunu ve aile etiketini içeren bir tablo döndürür.

    Görüntüler bu aşamada belleğe yüklenmez; sadece yollar listelenir.
    Sütunlar: path, family, label_id
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {root}\n"
            "Malimg'i indirip data/raw/ altına açın (README'ye bakın)."
        )

    rows = [
        {"path": str(p), "family": p.parent.name}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    if not rows:
        raise ValueError(f"{root} altında hiç görüntü bulunamadı.")

    df = pd.DataFrame(rows)
    families = sorted(df["family"].unique())
    df["label_id"] = df["family"].map({f: i for i, f in enumerate(families)})
    return df


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Her ailedeki örnek sayısını ve yüzdesini verir (çoktan aza sıralı)."""
    counts = df["family"].value_counts()
    return pd.DataFrame({
        "adet": counts,
        "yuzde": (counts / counts.sum() * 100).round(2),
    })


def image_stats(df: pd.DataFrame, sample_size: int | None = None,
                random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Görüntülerin genişlik, yükseklik, mod ve dosya boyutu bilgilerini çıkarır.

    sample_size verilirse sadece rastgele o kadar görüntüye bakılır (hız için).
    """
    sub = df if sample_size is None or sample_size >= len(df) else \
        df.sample(sample_size, random_state=random_state)

    records = []
    for _, row in sub.iterrows():
        with Image.open(row["path"]) as img:
            w, h = img.size
            mode = img.mode
        records.append({
            "path": row["path"],
            "family": row["family"],
            "width": w,
            "height": h,
            "mode": mode,
            "file_kb": Path(row["path"]).stat().st_size / 1024,
        })
    return pd.DataFrame(records)


def load_image(path: str | Path) -> np.ndarray:
    """Görüntüyü gri tonlamalı (uint8, 0-255) numpy dizisi olarak okur."""
    with Image.open(path) as img:
        return np.array(img.convert("L"))


if __name__ == "__main__":
    data = load_dataset_index()
    print(f"Toplam görüntü: {len(data)}")
    print(f"Aile sayısı   : {data['family'].nunique()}\n")
    print(class_distribution(data).to_string())
