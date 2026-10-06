# 4. Hafta: Özellik Çıkarımı (26 Ekim – 1 Kasım)

**Hedef:** 9.339 görüntünün her biri için 83 boyutlu özellik vektörü (59 LBP + 24 GLCM) çıkarılmış ve `data/processed/features.csv` dosyasına yazılmış olsun. Özellikler incelenmiş, rapordaki "Özellik Çıkarımı" bölümü yazılmış olsun.

**Ön koşul:** 3. haftanın sonunda `data/processed/images_256/index.csv` dosyası 9.339 satır içeriyor olmalı.

| Gün | Tarih | Deniz (LBP) | Berivan (GLCM) |
|---|---|---|---|
| 21–22 | 26–27 Eki | `src/features.py` içindeki LBP kısmını incele, Ojala (2002) ile karşılaştır. Neden `nri_uniform` seçtiğimizi ve neden 59 boyut çıktığını rapora yazabilecek kadar anla. Notebook 03, 1. bölüm. | `src/features.py` içindeki GLCM kısmını incele. 6 özelliğin formüllerini Haralick (1973) ve scikit-image dokümantasyonundan çıkar. Notebook 03, 2. bölüm. |
| 23 | 28 Eki | Tüm veri setinde özellik çıkarımını çalıştır (`python -m src.features --n-jobs -1`), süreyi not et. `features.csv` dosyasını Berivan'la paylaş. | `tests/test_features.py` dosyasını gözden geçir. Eksik gördüğün bir durum için en az bir test ekle. |
| 24 | 29 Eki | 🇹🇷 **Cumhuriyet Bayramı, tatil** | |
| 25 | 30 Eki | Notebook 03, 3. ve 4. bölümler: veri kontrolü (eksik veya sonsuz değer var mı) ve özelliklerin aileleri ayırt edip etmediği. | Notebook 03, 5. bölüm: GLCM korelasyonu. Birbiriyle çok yüksek korelasyonlu özellikleri listele. |
| 26 | 31 Eki | Rapor: **Yöntem – LBP** alt başlığı (formül ve `03_lbp_ornek.png` şekli). Berivan'ın pull request'ini incele. | Rapor: **Yöntem – GLCM** alt başlığı (formül tablosu ve `03_glcm_ornek.png` şekli). Deniz'in pull request'ini incele. |
| 27 | 1 Kas | **Toplantı 4**: `features.csv` kontrolü, notebook'taki notlar, 5. haftanın (train/test bölme ve ilk modeller) görev dağılımı. | **Toplantı 4** |

## Kontrol listesi (1 Kasım)
- [ ] `pytest -q` geçiyor (26 veya daha fazla test)
- [ ] `data/processed/features.csv` → **9.339 satır × 86 sütun** (path, family, label_id + 83 özellik)
- [ ] Eksik veya sonsuz değer yok
- [ ] `03_ozellik_cikarimi.ipynb` gerçek veriyle çalıştırıldı ve "Notlar" bölümü dolduruldu
- [ ] `reports/figures/03_*.png` grafikleri (5 adet) hazır
- [ ] Rapor: Yöntem → LBP ve GLCM alt başlıkları yazıldı

## Not
Formda "4–5. hafta: özellik çıkarımı ve train/test bölme" yazıyordu. Özellik çıkarımı 4. haftada biterse 5. haftada bölme ve ölçeklendirmeyle birlikte ilk modellere (SVM ve RF, varsayılan parametrelerle) başlanabilir. Böylece takvimde bir haftalık pay kazanılır.
