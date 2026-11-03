"""6. hafta: sonuçların birikmesi, CV kayıtları ve karşılaştırma aracı."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import compare, train  # noqa: E402
from src.features import feature_names  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Sahte features.csv ve tüm çıktıların geçici klasöre yazılması."""
    rng = np.random.default_rng(0)
    rows = []
    for k, fam in enumerate(["A", "B", "C"]):
        for i in range(40):
            vec = rng.normal(loc=k * 1.5, scale=1.0, size=83)
            rows.append({"path": f"{fam}/{i}.png", "family": fam, "label_id": k,
                         **dict(zip(feature_names(), vec))})
    csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    reports = tmp_path / "reports"
    monkeypatch.setattr(train, "REPORTS_DIR", reports)
    monkeypatch.setattr(compare, "REPORTS_DIR", reports)
    monkeypatch.setattr(train, "SPLIT_CSV", tmp_path / "split.csv")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "FIGURES_DIR", reports / "figures")
    # Testler hızlı olsun diye küçük ızgaralar
    monkeypatch.setitem(train.PARAM_GRIDS, "svm", {"clf__C": [1, 10], "clf__gamma": ["scale", 0.01]})
    monkeypatch.setitem(train.PARAM_GRIDS, "rf", {"clf__n_estimators": [20, 40]})
    monkeypatch.setitem(train.PARAM_GRIDS, "knn", {"clf__n_neighbors": [3, 5]})
    return csv, reports


def test_results_accumulate_across_runs(workspace):
    csv, reports = workspace
    train.run(["svm"], tune=False, features_csv=csv, n_jobs=1)
    train.run(["knn"], tune=False, features_csv=csv, n_jobs=1)
    res = pd.read_csv(reports / "sonuclar_varsayilan.csv")
    assert list(res["model"]) == ["SVM (RBF)", "KNN"]          # ikisi de duruyor, sıralı
    train.run(["svm"], tune=False, features_csv=csv, n_jobs=1)  # tekrar: çoğalmamalı
    assert len(pd.read_csv(reports / "sonuclar_varsayilan.csv")) == 2


def test_tune_saves_cv_and_family_files(workspace):
    csv, reports = workspace
    train.run(["svm"], tune=True, features_csv=csv, n_jobs=1)
    cv = pd.read_csv(reports / "cv_svm.csv")
    assert len(cv) == 4 and {"C", "gamma", "cv_f1_makro_ort", "sira"} <= set(cv.columns)
    fam = pd.read_csv(reports / "aile_f1_svm_ayarli.csv")
    assert set(fam["aile"]) == {"A", "B", "C"}
    pred = pd.read_csv(reports / "tahminler_svm_ayarli.csv")
    assert len(pred) == 24 and {"path", "family", "tahmin"} <= set(pred.columns)


def test_compare_end_to_end(workspace):
    csv, reports = workspace
    train.run(["svm", "rf", "knn"], tune=False, features_csv=csv, n_jobs=1)
    train.run(["svm", "rf", "knn"], tune=True, features_csv=csv, n_jobs=1)
    df = compare.main()
    assert len(df) == 3
    assert {"f1_makro_varsayilan", "f1_makro_ayarli", "f1_makro_fark"} <= set(df.columns)
    for name in ["06_varsayilan_vs_ayarli.png", "06_svm_izgara.png", "06_aile_f1_degisim.png"]:
        assert (reports / "figures" / name).exists(), name
    assert (reports / "karsilastirma.csv").exists()


def test_compare_needs_both_runs(workspace):
    csv, reports = workspace
    train.run(["knn"], tune=False, features_csv=csv, n_jobs=1)
    with pytest.raises(FileNotFoundError):
        compare.load_results()
