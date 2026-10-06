"""Ortamın doğru kurulduğunu kontrol eder: python scripts/check_env.py"""
import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PACKAGES = {
    "numpy": "numpy", "PIL": "pillow", "cv2": "opencv-python",
    "skimage": "scikit-image", "sklearn": "scikit-learn", "pandas": "pandas",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "joblib": "joblib",
    "streamlit": "streamlit",
}

print(f"Python {sys.version.split()[0]}", "(OK)" if sys.version_info[:2] == (3, 11)
      else "(uyarı: proje Python 3.11 için planlandı)")

missing = []
for module, pip_name in PACKAGES.items():
    try:
        m = importlib.import_module(module)
        print(f"  [OK]  {pip_name:<15} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"  [YOK] {pip_name}")
        missing.append(pip_name)

print()
if config.MALIMG_DIR.exists():
    n = sum(1 for p in config.MALIMG_DIR.rglob("*")
            if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    print(f"Malimg bulundu: {n} görüntü ({config.MALIMG_DIR})")
else:
    print(f"Malimg henüz yok: {config.MALIMG_DIR}")

if missing:
    print("\nEksik paketler için: pip install -r requirements.txt")
    sys.exit(1)
print("\nOrtam hazır.")
