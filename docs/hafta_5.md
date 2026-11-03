hafta_5
5. Hafta: İlk Modeller
Hedef: Aynı train/test bölmesi üzerinde SVM, Random Forest ve KNN'i varsayılan parametrelerle eğitmek ve ilk sonuç tablosunu çıkarmak.
Ön koşul: data/processed/features.csv (9.339 satır × 86 sütun)
Komut
.venv\Scripts\activate
python -m src.train
​
Süre: birkaç dakika veya daha az.
Ne yapıyor?
features.csv dosyasını okur.
Veriyi aile oranlarını koruyarak %80 eğitim, %20 test olarak böler (random_state=42). Bölmeyi data/processed/split.csv dosyasına yazar.
Her modeli bir Pipeline içinde eğitir: önce StandardScaler, sonra sınıflandırıcı. Ölçekleyici yalnızca eğitim verisiyle ayarlanır, böylece test verisinden bilgi sızmaz.
SVM: RBF çekirdek, class_weight="balanced"
Random Forest: class_weight="balanced"
KNN: k=5
Test setinde doğruluk, makro kesinlik, makro duyarlılık ve makro F1 hesaplar.
Çıktılar
Dosya
İçerik
reports/sonuclar_varsayilan.csv
3 modelin karşılaştırma tablosu (rapora girecek)
reports/<model>_varsayilan_rapor.txt
Aile bazında kesinlik, duyarlılık ve F1
reports/figures/05_cm_<model>_varsayilan.png
Karışıklık matrisleri
models/<model>_varsayilan.joblib
Eğitilmiş modeller
Görev dağılımı
Kişi
Görev
Deniz
src/train.py dosyasını oku: split_data ve build_models fonksiyonları. Neden Pipeline kullandığımızı rapora yaz. SVM sonuçlarını yorumla.
Berivan
Random Forest ve KNN sonuçlarını yorumla. Karışıklık matrislerinde en çok karışan aile çiftlerini listele.
İkiniz
Sonuç tablosunu notlar.md dosyasına ve rapora ekleyin. Makro F1 neden doğruluktan düşük, tartışın.
Sorulacak sorular (rapora girecek)
Hangi model en iyi makro F1'i verdi?
Doğruluk ile makro F1 arasındaki fark ne kadar? Fark büyükse az örnekli aileler zorlanıyor demektir.
Hangi aileler birbirine karışıyor? 2. haftada tahmin ettiğimiz Lolyda, C2LOP ve Swizzor alt aileleri mi?
Sonraki adım (6.–8. hafta)
python -m src.train --tune
​
Bu komut GridSearchCV ve 5 katlı StratifiedKFold ile her model için en iyi parametreleri arar ve çıktıları _ayarli ekiyle kaydeder. SVM'nin arama süresi uzun olabilir. Tek modeli denemek için: python -m src.train --tune --models knn