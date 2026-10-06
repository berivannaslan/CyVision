"""Proje genelinde kullanılan sabitler ve dosya yolları."""
from pathlib import Path

# Klasörler
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # Malimg buraya açılacak
PROCESSED_DIR = DATA_DIR / "processed"  # features.csv vb.
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Malimg klasörü (data/raw altında aileler alt klasör olarak durur)
MALIMG_DIR = RAW_DIR / "malimg_paths"

# Görüntü ön işleme
IMG_SIZE = (256, 256)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Veri bölme ve tekrarlanabilirlik
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Özellik çıkarımı (3.-5. haftalarda kullanılacak)
LBP_P = 8
LBP_R = 1
LBP_METHOD = "nri_uniform"   # P=8 için 59 bin
GLCM_DISTANCES = [1]
GLCM_ANGLES_DEG = [0, 45, 90, 135]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
