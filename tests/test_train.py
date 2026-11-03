"""Model eğitimi testleri: sahte, ayrılabilir bir özellik veri setiyle."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import train  # noqa: E402
from src.features import feature_names  # noqa: E402


@pytest.fixture
def fake_features(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for k, fam in enumerate(["A", "B", "C"]):
        for i in range(30):
            vec = rng.normal(loc=k * 3, scale=0.5, size=83)  # aileler net ayrışıyor
            rows.append({"path": f"{fam}/{i}.png", "family": fam, "label_id": k,
                         **dict(zip(feature_names(), vec))})
    csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    monkeypatch.setattr(train, "SPLIT_CSV", tmp_path / "split.csv")
    return csv


def test_split_is_stratified(fake_features):
    from src.features import load_features
    X, y, meta = load_features(fake_features)
    X_tr, X_te, y_tr, y_te = train.split_data(X, y, meta)
    assert len(y_te) == 18 and len(y_tr) == 72
    assert (y_te.value_counts() == 6).all()          # her aileden %20
    split = pd.read_csv(train.SPLIT_CSV)
    assert (split["split"] == "test").sum() == 18


def test_baseline_models(fake_features):
    res = train.run(["svm", "rf", "knn"], tune=False, features_csv=fake_features, save=False)
    assert list(res["model"]) == ["SVM (RBF)", "Random Forest", "KNN"]
    assert (res["dogruluk"] > 0.95).all()             # ayrılabilir veride neredeyse kusursuz


def test_tuning_small(fake_features, monkeypatch):
    monkeypatch.setitem(train.PARAM_GRIDS, "knn", {"clf__n_neighbors": [1, 3]})
    res = train.run(["knn"], tune=True, features_csv=fake_features, n_jobs=1, save=False)
    assert "n_neighbors" in res["parametreler"].iloc[0]


def test_pipeline_has_scaler():
    for model in train.build_models().values():
        assert model.steps[0][0] == "scaler"
