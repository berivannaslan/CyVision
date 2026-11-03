# 6. Hafta: Parametre Ayarı ve Model Karşılaştırması

**Hedef:** Üç modelin en iyi parametrelerini, yalnızca eğitim verisi üzerinde 5 katlı çapraz doğrulamayla bulmak. Ardından varsayılan sonuçlarla karşılaştırıp Swizzor/C2LOP karışıklığının azalıp azalmadığını görmek.

**Ön koşul:** `data/processed/features.csv` ve 5. haftanın varsayılan sonuçları (`reports/sonuclar_varsayilan.csv`).

## Bu hafta eklenenler

| Dosya | Ne yapıyor |
|---|---|
| `src/train.py` (güncellendi) | Sonuçlar artık birikiyor, modelleri tek tek eğitsen de dosyada hepsi kalıyor. Ayrıca CV ayrıntılarını, aile bazında F1'i ve her test örneğinin tahminini kaydediyor. |
| `src/compare.py` (yeni) | Varsayılan ve ayarlı sonuçları karşılaştırıyor, 3 grafik ve 2 tablo üretiyor |
| `notebooks/04_parametre_ayari_ve_karsilastirma.ipynb` (yeni) | Sonuçları yorumluyor, Swizzor görüntülerini yan yana koyuyor, hatalı örnekleri gösteriyor |
| `tests/test_compare.py` (yeni) | 4 yeni test (toplam 34) |

## Gün gün plan

### Gün 1: Hazırlık ve varsayılan sonuçları yenileme (Deniz)
- [ ] Yeni dosyaları projeye ekle: `src/train.py`, `src/compare.py`, `tests/test_compare.py`, `notebooks/04_...ipynb`, `docs/hafta_6.md`.
- [ ] Testleri çalıştır: `pytest -q` → **34 passed**.
- [ ] Varsayılan modelleri yeniden eğit. Yeni çıktı dosyaları (aile F1, tahminler) bu adımda oluşuyor:
  ```powershell
  python -m src.train
  ```
  Sonuçlar 5. haftayla aynı olmalı (RF %98,45). Bölme aynı tohumla (42) yapıldığı için değişmez.

### Gün 2–3: Parametre ayarı (Deniz)
Modelleri hızlıdan yavaşa doğru tek tek çalıştır. Sonuçlar artık birikiyor:
```powershell
Measure-Command { python -m src.train --tune --models knn }   # 20 kombinasyon, ~1 dk
Measure-Command { python -m src.train --tune --models svm }   # 16 kombinasyon, birkaç dk
Measure-Command { python -m src.train --tune --models rf }    # 18 kombinasyon, en uzunu (5–10 dk)
```
- [ ] Her birinin ekrandaki "En iyi parametreler" ve süre satırını `notlar.md` dosyasına yaz.
- [ ] `reports\cv_svm.csv`, `cv_rf.csv` ve `cv_knn.csv` dosyalarının oluştuğunu kontrol et.

Neyin denendiği:

| Model | Parametre | Denenen değerler |
|---|---|---|
| SVM | C | 1, 10, 100, 1000 |
| SVM | gamma | scale, 0.001, 0.01, 0.1 |
| RF | n_estimators | 100, 300, 500 |
| RF | max_depth | sınırsız, 20, 40 |
| RF | min_samples_leaf | 1, 2 |
| KNN | n_neighbors | 1, 3, 5, 7, 9 |
| KNN | weights | uniform, distance |
| KNN | metric | euclidean, manhattan |

### Gün 4: Karşılaştırma (Deniz, Berivan)
```powershell
python -m src.compare
```
Üretilen dosyalar:
- `reports\karsilastirma.csv`: önce/sonra tablosu, rapora girecek
- `reports\figures\06_varsayilan_vs_ayarli.png`: makro F1, önce/sonra
- `reports\figures\06_svm_izgara.png`: SVM C × gamma ısı haritası
- `reports\figures\06_aile_f1_degisim.png`: en iyi modelde aile bazında değişim
- `reports\hatali_ciftler_<model>.csv`: en çok karışan aile çiftleri

### Gün 5: Notebook ve görsel kanıt (Berivan)
```powershell
jupyter notebook notebooks\04_parametre_ayari_ve_karsilastirma.ipynb
```
- [ ] **Restart & Run All** yap.
- [ ] 6. bölüm: Swizzor.gen!E ve gen!I örnek görüntüleri ile ortalama görüntü farkı. Bu fark, iki farklı ailenin farkıyla karşılaştırılıyor.
- [ ] 7. bölüm: Yanlış sınıflandırılan örnekler.
- [ ] "Notlar" hücresindeki 5 soruyu cevapla.

### Gün 6: Rapor (Deniz, Berivan)
- [ ] Bitirme raporuna "Parametre ayarı" alt bölümü ekle: denenen ızgara (yukarıdaki tablo), en iyi parametreler, önce/sonra tablosu, `06_varsayilan_vs_ayarli.png` ve `06_svm_izgara.png` grafikleri.
- [ ] "Hata analizi" bölümünü güncelle: ayardan sonra Swizzor/C2LOP karışıklığı ve `06_swizzor_ortalama_fark.png` grafiği.

### Gün 7: Toplantı ve kapanış
- [ ] Sonuçları birlikte yorumlayın, raporda "en iyi model" olarak hangisinin sunulacağına karar verin.
- [ ] Commit ve push:
  ```powershell
  git add .
  git commit -m "6. hafta: GridSearchCV ile parametre ayarı ve karşılaştırma"
  git push
  ```

## Dikkat edilecekler
- **Test seti parametre seçiminde kullanılmadı.** GridSearchCV yalnızca eğitim setinin (7.471 örnek) içinde 5 katlı çapraz doğrulama yapıyor. Raporda bunu açıkça yazın; hocanın en çok soracağı konulardan biri.
- **Ayarlı model her zaman daha iyi çıkmayabilir.** CV'de en iyi olan parametre tek bir test setinde varsayılandan biraz düşük çıkabilir. Fark 0,005'in altındaysa bu rastlantısal bir farktır, model kötüleşti anlamına gelmez. Böyle olursa raporda dürüstçe yazın ve CV sonucunu (ortalama ± standart sapma) da verin.
- **Süre de bir ölçüttür.** Benzer F1 veren iki modelden daha hızlı olanı, Streamlit arayüzü (11. hafta) için daha uygun olur.

## Kontrol listesi
- [ ] `pytest -q` → 34 passed
- [ ] `reports\sonuclar_ayarli.csv` dosyasında 3 model var
- [ ] `reports\cv_*.csv` → 3 dosya
- [ ] `reports\karsilastirma.csv` ve 06_* grafikleri (6 adet) oluştu
- [ ] Notebook 04'ün notları dolu
- [ ] Rapora "Parametre ayarı" bölümü eklendi
