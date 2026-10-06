"""Gerçek veri olmadan, sahte küçük bir veri setiyle data_loader'ı test eder.

Çalıştırma: pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_loader  # noqa: E402


@pytest.fixture
def fake_malimg(tmp_path):
    rng = np.random.default_rng(0)
    for family, n in {"Allaple.A": 5, "Yuner.A": 3, "Lolyda.AA1": 2}.items():
        d = tmp_path / "malimg_paths" / family
        d.mkdir(parents=True)
        for i in range(n):
            arr = rng.integers(0, 256, size=(64 + i * 8, 64), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")
    return tmp_path / "malimg_paths"


def test_index(fake_malimg):
    df = data_loader.load_dataset_index(fake_malimg)
    assert len(df) == 10
    assert df["family"].nunique() == 3
    assert set(df["label_id"]) == {0, 1, 2}


def test_distribution(fake_malimg):
    dist = data_loader.class_distribution(data_loader.load_dataset_index(fake_malimg))
    assert dist.index[0] == "Allaple.A"
    assert dist["adet"].sum() == 10
    assert abs(dist["yuzde"].sum() - 100) < 0.1


def test_stats_and_load(fake_malimg):
    df = data_loader.load_dataset_index(fake_malimg)
    stats = data_loader.image_stats(df)
    assert (stats["width"] == 64).all()
    img = data_loader.load_image(df["path"].iloc[0])
    assert img.dtype == np.uint8 and img.ndim == 2


def test_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_loader.load_dataset_index(tmp_path / "yok")
