# Okuma Listesi (1.–2. hafta)

Bu kaynaklar, rapordaki "Literatür" ve "Yöntem" bölümlerinin temelini oluşturacak. Her birini okuduktan sonra altına 3–5 cümlelik bir özet yazın: ne yapmışlar, hangi veri setini kullanmışlar, ne sonuç almışlar ve bizim projeyle bağlantısı ne.

> Künyeleri rapora koymadan önce Google Scholar'dan doğrulayın (yıl, sayfa, DOI).

## Zararlı yazılımı görüntü olarak inceleme
1. **Nataraj, L., Karthikeyan, S., Jacob, G., Manjunath, B. S. (2011).** *Malware Images: Visualization and Automatic Classification.* VizSec 2011.
   Malimg veri setini ve binary'den görüntüye dönüştürme yöntemini tanıtan makale. Kullandığımız genişlik tablosu bu makaleden geliyor. **Önce bunu okuyun.**
   - Özet:
2. **Luo, J.-S., Lo, D. C.-T. (2017).** *Binary Malware Image Classification Using Machine Learning with Local Binary Pattern.* IEEE BigData 2017.
   Doğrudan bizim yaklaşımımıza yakın bir çalışma (LBP ile malware görüntüsü sınıflandırma).
   - Özet:
3. **Kancherla, K., Mukkamala, S. (2013).** *Image Visualization Based Malware Detection.* IEEE CICS 2013.
   - Özet:

## Doku özellikleri
4. **Ojala, T., Pietikäinen, M., Mäenpää, T. (2002).** *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns.* IEEE TPAMI, 24(7).
   LBP'nin temel makalesi. "uniform" örüntü kavramı burada tanımlanıyor.
   - Özet:
5. **Haralick, R. M., Shanmugam, K., Dinstein, I. (1973).** *Textural Features for Image Classification.* IEEE Trans. Systems, Man, and Cybernetics, SMC-3(6).
   GLCM ve kontrast, homojenlik, enerji, korelasyon gibi özelliklerin kaynağı.
   - Özet:

## Derin öğrenme ile karşılaştırma (raporda "neden CNN kullanmadık" kısmı için)
6. **Kalash, M., Rochan, M., Mohammed, N., Bruce, N. D. B., Wang, Y., Iqbal, F. (2018).** *Malware Classification with Deep Convolutional Neural Networks.* NTMS 2018.
   - Özet:
7. **Gibert, D., Mateu, C., Planes, J. (2020).** *The Rise of Machine Learning for Detection and Classification of Malware: Research Developments, Trends and Challenges.* Journal of Network and Computer Applications, 153.
   Alanın genel derlemesi; giriş bölümü için faydalı.
   - Özet:

## Kütüphane dokümantasyonu (4. haftaya hazırlık)
- scikit-image: `skimage.feature.local_binary_pattern`, `graycomatrix`, `graycoprops`
- scikit-learn: `SVC`, `RandomForestClassifier`, `KNeighborsClassifier`, `GridSearchCV`, `StratifiedKFold`
