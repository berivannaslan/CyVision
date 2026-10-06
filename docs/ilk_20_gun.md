# İlk 20 Gün: Görev Dağılımı (6–25 Ekim)

**Hedef:** 25 Ekim'e kadar veri seti hazır ve incelenmiş, dönüştürme ve ön işleme kodu çalışıyor, 9.339 görüntünün hepsi 256×256'ya getirilmiş olsun. Rapor taslağının ilk iki bölümü de yazılmış olsun.

## 1. Hafta (6–11 Ekim): Kurulum ve veri seti

| Gün | Tarih | Deniz | Berivan |
|---|---|---|---|
| 1–2 | 6–7 Eki | GitHub deposunu aç, kodu push et, Berivan'ı ekle. Her görev için issue aç. | Python 3.11 ve sanal ortamı kur. `check_env.py` ve `pytest` sorunsuz geçsin. |
| 3–5 | 8–10 Eki | Malimg'i indir ve `data/raw/malimg_paths/` altına aç. `python -m src.data_loader` komutu **9.339 görüntü / 25 aile** göstermeli. Veriyi Berivan'la paylaş (Drive veya USB). | Nataraj (2011) ve Luo & Lo (2017) makalelerini oku, `docs/literatur.md` dosyasına özetlerini yaz. |
| 6 | 11 Eki | **Toplantı 1** (30 dk): kurulum ve veri seti tamam mı? | **Toplantı 1** |

**1. haftanın çıktısı:** İki bilgisayarda da çalışan ortam, veri seti, iki makale özeti.

## 2. Hafta (12–18 Ekim): Veri inceleme ve literatür

| Gün | Tarih | Deniz | Berivan |
|---|---|---|---|
| 7–9 | 12–14 Eki | Ojala (2002, LBP) ve Haralick (1973, GLCM) makalelerini oku. scikit-image'da LBP ve GLCM'yi tek bir görüntüde dene (4. haftaya hazırlık). | `01_veri_inceleme.ipynb`'yi gerçek veriyle çalıştır. Grafikleri incele ve "Notlar" bölümünü doldur: dengesizlik oranı ve birbirine benzeyen aileler. |
| 10–12 | 15–17 Eki | Rapor taslağı: **Giriş ve Literatür** (2–3 sayfa). | Rapor taslağı: **Veri Seti** bölümü (1–2 sayfa, sınıf dağılımı ve örnek görüntü grafikleriyle). |
| 13 | 18 Eki | **Toplantı 2**: notebook ve taslaklar `main` dalına alınır. | **Toplantı 2** |

**2. haftanın çıktısı:** Doldurulmuş veri inceleme notebook'u, `reports/figures/01_*.png` grafikleri, rapordaki ilk iki bölümün taslağı.

## 3. Hafta (19–25 Ekim): Dönüştürme ve ön işleme

| Gün | Tarih | Deniz | Berivan |
|---|---|---|---|
| 14–16 | 19–21 Eki | `src/binary_to_image.py`: koddaki genişlik tablosunu makaleyle karşılaştır. **Zararsız** dosyalarla dene (`notepad.exe`, `calc.exe`, `python.exe`). `02_...ipynb` notebook'unun 1. ve 2. bölümleri. | `src/preprocess.py`: tüm veri setini işle (`python -m src.preprocess`), süreyi ölç. `02_...ipynb` notebook'unun 3. ve 4. bölümleri. |
| 17–18 | 22–23 Eki | Berivan'ın pull request'ini incele (code review). | Deniz'in pull request'ini incele (code review). |
| 19 | 24 Eki | Rapor: **Yöntem – Dönüştürme** paragrafı ve görseli. | Rapor: **Yöntem – Ön İşleme** paragrafı. README'deki durum tablosunu güncelle. |
| 20 | 25 Eki | **Toplantı 3**: 3. haftayı kapat, 4. haftanın (LBP ve GLCM) görevlerini dağıt. | **Toplantı 3** |

**3. haftanın çıktısı:** `data/processed/images_256/` klasöründe 9.339 adet 256×256 görüntü ve `index.csv`, `reports/figures/02_*.png` grafikleri, rapordaki Yöntem bölümünün ilk kısmı.

## Kontrol listesi (25 Ekim)
- [ ] `pytest -q` iki bilgisayarda da geçiyor
- [ ] Malimg: 9.339 görüntü, 25 aile
- [ ] `01_veri_inceleme.ipynb` gerçek veriyle çalıştırıldı, notlar yazıldı
- [ ] `02_donusturme_ve_onisleme.ipynb` çalıştırıldı
- [ ] `data/processed/images_256/index.csv` → 9.339 satır
- [ ] En az 5 makalenin özeti `docs/literatur.md` dosyasında
- [ ] Rapor taslağı: Giriş, Literatür, Veri Seti, Yöntem (dönüştürme ve ön işleme)
