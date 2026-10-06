"""LBP + GLCM özellik çıkarımı testleri."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import features as F  # noqa: E402

rng = np.random.default_rng(0)


def rand_img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (256, 256), dtype=np.uint8)


def test_dimensions():
    assert F.N_LBP_BINS == 59
    assert len(F.glcm_feature_names()) == 24
    assert len(F.feature_names()) == 83
    assert len(set(F.feature_names())) == 83  # isimler benzersiz


def test_lbp_histogram():
    h = F.lbp_features(rand_img())
    assert h.shape == (59,)
    assert np.isclose(h.sum(), 1.0)
    assert (h >= 0).all()


def test_lbp_flat_image():
    # Düz görüntüde iç pikseller tek bir örüntü verir (kenar pikselleri hariç)
    h = F.lbp_features(np.full((64, 64), 100, np.uint8))
    assert h.argmax() == 57 and h.max() > 0.9


def test_glcm_values():
    g = F.glcm_features(rand_img())
    assert g.shape == (24,)
    names = F.glcm_feature_names()
    vals = dict(zip(names, g))
    # Homojenlik, enerji ve ASM 0-1 aralığında olmalı
    for n, v in vals.items():
        if any(k in n for k in ("homogeneity", "energy", "ASM")):
            assert 0 <= v <= 1, n
    # ASM = enerji^2
    assert np.isclose(vals["glcm_ASM_1_0"], vals["glcm_energy_1_0"] ** 2)


def test_glcm_flat_vs_noise():
    flat = dict(zip(F.glcm_feature_names(), F.glcm_features(np.full((64, 64), 50, np.uint8))))
    noise = dict(zip(F.glcm_feature_names(), F.glcm_features(rand_img())))
    assert flat["glcm_contrast_1_0"] == 0
    assert noise["glcm_contrast_1_0"] > 1000
    assert flat["glcm_homogeneity_1_0"] > noise["glcm_homogeneity_1_0"]


def test_extract_features():
    v = F.extract_features(rand_img())
    assert v.shape == (83,) and np.isfinite(v).all()
    # deterministik
    assert np.array_equal(v, F.extract_features(rand_img()))
    with pytest.raises(ValueError):
        F.extract_features(rand_img().astype(np.float32))


def test_extract_dataset(tmp_path):
    rows = []
    for fam in ["A", "B"]:
        (tmp_path / fam).mkdir()
        for i in range(3):
            p = tmp_path / fam / f"{i}.png"
            Image.fromarray(rand_img(i + (10 if fam == "B" else 0))).save(p)
            rows.append({"path": str(p), "family": fam, "label_id": int(fam == "B")})
    pd.DataFrame(rows).to_csv(tmp_path / "index.csv", index=False)

    out = F.extract_dataset(tmp_path / "index.csv", tmp_path / "features.csv")
    assert out.shape == (6, 3 + 83)
    X, y, meta = F.load_features(tmp_path / "features.csv")
    assert X.shape == (6, 83) and list(y) == ["A"] * 3 + ["B"] * 3

    par = F.extract_dataset(tmp_path / "index.csv", tmp_path / "f2.csv", n_jobs=2)
    assert np.allclose(par[F.feature_names()].values, out[F.feature_names()].values)
