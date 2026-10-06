"""Bir dosyanın byte'larını gri tonlamalı görüntüye çevirir (Nataraj vd., 2011 yöntemi).

Dosya hiçbir şekilde ÇALIŞTIRILMAZ; yalnızca ham byte'ları okunur.
Her byte (0-255) bir piksel olur; genişlik dosya boyutuna göre seçilir,
yükseklik = byte sayısı / genişlik.

Kullanım (yalnızca zararsız dosyalarla):
    python -m src.binary_to_image C:\\Windows\\System32\\notepad.exe C:\\Windows\\System32\\calc.exe -o reports/figures
    python -m src.binary_to_image klasor/ -o data/processed/converted
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Nataraj vd. (2011) tablosu: (üst sınır KB, genişlik piksel)
WIDTH_TABLE = [
    (10, 32),
    (30, 64),
    (60, 128),
    (100, 256),
    (200, 384),
    (500, 512),
    (1000, 768),
    (float("inf"), 1024),
]


def choose_width(size_bytes: int) -> int:
    """Dosya boyutuna göre görüntü genişliğini seçer."""
    size_kb = size_bytes / 1024
    for limit_kb, width in WIDTH_TABLE:
        if size_kb < limit_kb:
            return width
    return WIDTH_TABLE[-1][1]


def bytes_to_array(data: bytes, width: int | None = None, pad: bool = True) -> np.ndarray:
    """Byte dizisini 2B uint8 matrise çevirir.

    pad=True ise son satır sıfırlarla tamamlanır; False ise yarım kalan satır atılır
    (orijinal makaledeki gibi).
    """
    if len(data) == 0:
        raise ValueError("Boş dosya görüntüye çevrilemez.")
    arr = np.frombuffer(data, dtype=np.uint8)
    width = width or choose_width(len(arr))

    rem = len(arr) % width
    if rem:
        if pad:
            arr = np.concatenate([arr, np.zeros(width - rem, dtype=np.uint8)])
        else:
            arr = arr[: len(arr) - rem]
    if len(arr) == 0:  # pad=False ve dosya tek satırdan kısa
        raise ValueError("Dosya seçilen genişlikten kısa; pad=True kullanın.")
    return arr.reshape(-1, width)


def file_to_image(path: str | Path, width: int | None = None, pad: bool = True) -> Image.Image:
    """Dosyayı okuyup gri tonlamalı PIL görüntüsü döndürür."""
    data = Path(path).read_bytes()
    return Image.fromarray(bytes_to_array(data, width, pad), mode="L")


def convert(path: str | Path, out_dir: str | Path, width: int | None = None) -> Path:
    """Tek dosyayı dönüştürüp <out_dir>/<dosya_adı>.png olarak kaydeder."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.name}.png"
    file_to_image(path, width).save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Dosya(lar)ı gri tonlamalı görüntüye çevirir.")
    ap.add_argument("inputs", nargs="+", help="Bir veya daha fazla dosya/klasör")
    ap.add_argument("-o", "--out", default="data/processed/converted", help="Çıktı klasörü")
    ap.add_argument("-w", "--width", type=int, default=None, help="Sabit genişlik (varsayılan: tablo)")
    args = ap.parse_args()

    files = []
    for item in map(Path, args.inputs):
        files += [p for p in item.rglob("*") if p.is_file()] if item.is_dir() else [item]
    for f in files:
        try:
            out = convert(f, args.out, args.width)
            with Image.open(out) as im:
                print(f"{f.name:<40} {f.stat().st_size / 1024:>8.1f} KB -> {im.size[0]}x{im.size[1]}")
        except (OSError, ValueError) as e:
            print(f"{f.name:<40} atlandı: {e}")


if __name__ == "__main__":
    main()
