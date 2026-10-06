# Görüntü İşleme ve Makine Öğrenmesi ile Statik Zararlı Yazılım Sınıflandırma

Tekirdağ Namık Kemal Üniversitesi, Bilgisayar Mühendisliği, Yazılım Temelli Proje Geliştirme I

**Ekip:** Deniz Özçelik · Berivan Aslan

Zararlı yazılım dosyalarının byte'larını gri tonlamalı görüntülere dönüştürüp bu görüntülerden **LBP** ve **GLCM** doku özellikleri çıkarıyoruz. Ardından **SVM**, **Random Forest** ve **KNN** ile Malimg veri setindeki 25 aileyi sınıflandırıyoruz.

## Klasör yapısı

```
malware-image-classification/
├── config.py              # Sabitler ve yollar (görüntü boyutu, LBP/GLCM parametreleri...)
├── requirements.txt
├── data/
│   ├── raw/               # Malimg buraya açılır (git'e girmez)
│   └── processed/         # features.csv vb. (git'e girmez)
├── models/                # Eğitilmiş .joblib modeller (git'e girmez)
├── docs/
│   ├── ilk_20_gun.md      # 6-25 Ekim görev dağılımı
│   └── literatur.md       # Okuma listesi ve özetler
├── notebooks/
│   ├── 01_veri_inceleme.ipynb
│   └── 02_donusturme_ve_onisleme.ipynb
├── reports/figures/       # Rapora girecek grafikler
├── scripts/
│   └── check_env.py       # Ortam kontrolü
├── src/
│   ├── data_loader.py     # Veri setini listeleme, sınıf dağılımı, görüntü okuma
│   ├── binary_to_image.py # Dosya byte'larını gri görüntüye çevirme (Nataraj 2011)
│   └── preprocess.py      # Gri okuma + 256x256, tüm veri setini işleme
└── tests/
    ├── test_data_loader.py
    └── test_preprocess.py
```

## Kurulum (Windows)

```powershell
# 1) Python 3.11 ile sanal ortam
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2) Paketler
pip install -r requirements.txt

# 3) Kontrol
python scripts\check_env.py
pytest -q
```

## Veri seti

1. Malimg veri setini indirin. Kaggle'da "Malimg dataset" diye aratınca bulabilirsiniz. Her ailenin kendi klasöründe PNG görüntüler olduğu sürümü seçin.
2. Zip dosyasını `data/raw/` altına açın. Beklenen yapı şöyle:
   ```
   data/raw/malimg_paths/Adialer.C/*.png
   data/raw/malimg_paths/Agent.FYI/*.png
   ...
   ```
   Klasörün adı farklıysa `config.py` içindeki `MALIMG_DIR` değerini değiştirin.
3. Doğrulayın: `python -m src.data_loader` komutu 9.339 görüntü ve 25 aile göstermeli.

> Malimg yalnızca görüntülerden oluşur ve içinde çalıştırılabilir zararlı dosya yoktur. `.gitignore`, `.exe`, `.dll` ve `.bin` dosyalarının depoya girmesini engeller.

## Veri inceleme

```powershell
jupyter notebook notebooks\01_veri_inceleme.ipynb
```

Grafikler `reports/figures/` klasörüne kaydedilir.

## Dönüştürme ve ön işleme

```powershell
# Zararsız bir dosyayı görüntüye çevir (dosya ÇALIŞTIRILMAZ, sadece byte'ları okunur)
python -m src.binary_to_image C:\Windows\System32\notepad.exe -o reports\figures

# Tüm Malimg'i 256x256'ya getir -> data/processed/images_256/
python -m src.preprocess
```

## Zaman planı

| Hafta | İş paketi | Durum |
|---|---|---|
| 1–2 | Hazırlık, veri seti, veri inceleme | 🟡 Devam ediyor |
| 3 | Binary'den görüntüye dönüştürme, ön işleme (256×256) | 🟡 Kod hazır, gerçek veriyle çalıştırılacak |
| 4–5 | LBP + GLCM ile 83 boyutlu özellik vektörü, train/test bölme | ⬜ |
| 6–8 | SVM, RF, KNN + GridSearchCV, ara rapor | ⬜ |
| 9–10 | Değerlendirme, karışıklık matrisleri, özellik önemi | ⬜ |
| 11 | Streamlit arayüzü | ⬜ |
| 12–13 | Bütünleştirme, final raporu, sunum | ⬜ |

## Git ile çalışma

- `main` dalı her zaman çalışır durumda tutulur.
- Her iş paketi için ayrı bir dal açılır (ör. `feature/lbp`, `feature/glcm`) ve iş bitince pull request ile birleştirilir.
- Her iş paketi GitHub'da bir issue olarak açılır ve sorumlusu atanır.
