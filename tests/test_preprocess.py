"""binary_to_image ve preprocess modüllerinin testleri (gerçek veri gerekmez)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import binary_to_image as b2i  # noqa: E402
from src import preprocess  # noqa: E402
from src.data_loader import load_dataset_index  # noqa: E402


# ---------- binary_to_image ----------
@pytest.mark.parametrize("kb,width", [(5, 32), (20, 64), (50, 128), (80, 256),
                                      (150, 384), (300, 512), (700, 768), (2000, 1024)])
def test_width_table(kb, width):
    assert b2i.choose_width(kb * 1024) == width


def test_bytes_preserved():
    data = bytes(range(256)) * 4  # 1024 byte
    arr = b2i.bytes_to_array(data, width=32)
    assert arr.shape == (32, 32)
    assert arr.ravel().tobytes() == data  # byte'lar birebir piksel olmalı


def test_padding():
    arr = b2i.bytes_to_array(b"\x01" * 70, width=32)
    assert arr.shape == (3, 32)
    assert arr[-1, 6:].sum() == 0  # dolgu sıfır
    assert b2i.bytes_to_array(b"\x01" * 70, width=32, pad=False).shape == (2, 32)


def test_empty_file():
    with pytest.raises(ValueError):
        b2i.bytes_to_array(b"")


def test_convert_file(tmp_path):
    f = tmp_path / "ornek.bin"
    f.write_bytes(np.random.default_rng(0).integers(0, 256, 40_000, dtype=np.uint8).tobytes())
    out = b2i.convert(f, tmp_path / "out")
    with Image.open(out) as im:
        assert im.mode == "L" and im.size[0] == 128


# ---------- preprocess ----------
@pytest.fixture
def fake_malimg(tmp_path):
    rng = np.random.default_rng(0)
    for fam, n in {"Allaple.A": 3, "Yuner.A": 2}.items():
        d = tmp_path / "malimg" / fam
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 256, (300 + 50 * i, 512), dtype=np.uint8)).save(d / f"{i}.png")
    # küçük görüntü (büyütme yolu)
    Image.fromarray(rng.integers(0, 256, (100, 64), dtype=np.uint8)).save(tmp_path / "malimg" / "Yuner.A" / "kucuk.png")
    return tmp_path / "malimg"


def test_preprocess_image(fake_malimg):
    for p in fake_malimg.rglob("*.png"):
        img = preprocess.preprocess_image(p)
        assert img.shape == (256, 256) and img.dtype == np.uint8


def test_unicode_path(tmp_path):
    p = tmp_path / "çalışma_ğüş" / "görüntü.png"
    p.parent.mkdir()
    Image.fromarray(np.full((50, 50), 7, np.uint8)).save(p)
    assert preprocess.preprocess_image(p).shape == (256, 256)


def test_preprocess_dataset(fake_malimg, tmp_path):
    df = load_dataset_index(fake_malimg)
    out_dir = tmp_path / "out"
    res = preprocess.preprocess_dataset(df, out_dir)
    assert len(res) == 6
    assert (out_dir / "index.csv").exists()
    assert {p.parent.name for p in out_dir.rglob("*.png")} == {"Allaple.A", "Yuner.A"}
    # ikinci çalıştırma mevcut dosyaları atlar ama aynı sonucu verir
    assert len(preprocess.preprocess_dataset(df, out_dir)) == 6
