# 🚗 Automobile Price Prediction (Otomobil Fiyat Tahmin Modeli)

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-success)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange?logo=scikit-learn)

Bu proje, bir gözetimli öğrenme (supervised learning) uygulaması olarak otomobillerin teknik özelliklerine ve donanımlarına bakarak piyasa fiyatlarını (price) tahmin etmeyi amaçlamaktadır. Proje kapsamında kapsamlı bir keşifçi veri analizi (EDA), özellik mühendisliği (feature engineering) ve model optimizasyonu gerçekleştirilmiştir.

## 📊 Veri Ön İşleme (Preprocessing) Adımları

Veri seti doğrudan algoritmaya verilmeden önce istatistiksel metotlarla temizlenmiş ve makine öğrenmesine hazır hale getirilmiştir:
* **Eksik Veri Yönetimi:** Hedef değişken (`price`) eksik olan satırlar veri setinden çıkarılmış, sayısal bağımsız değişkenler medyan (ortanca) değerlerle, kategorik değişkenler ise mod (en çok tekrar eden) değerlerle doldurulmuştur.
* **Kategorik Kodlama (Encoding):** * Silindir sayısı ve kapı sayısı gibi hiyerarşik yapıdaki veriler için `OrdinalEncoder` kullanılmıştır.
  * İkili (binary) ve nominal kategorik veriler için Dummy değişken tuzağına düşmemek adına `OneHotEncoder(drop="first")` uygulanmıştır.
* **Ölçeklendirme:** Tüm bağımsız değişkenler `StandardScaler` ile standartlaştırılmıştır.

## 🤖 Kullanılan Makine Öğrenmesi Modelleri

Projeye dahil edilen ve performansları karşılaştırılan regresyon modelleri şunlardır:
* Linear Regression, RidgeCV, LassoCV
* K-Neighbors Regressor
* Decision Tree & Random Forest Regressor
* AdaBoost & Gradient Boosting Regressor
* XGBoost & LightGBM Regressor

Modellerin başarısı sadece test verisiyle değil, aşırı öğrenme (overfitting) durumunu tespit edebilmek amacıyla **Eğitim ve Test R2 skorları arasındaki fark** baz alınarak değerlendirilmiştir.

## ⚙️ Hiperparametre Optimizasyonu

Karşılaştırma testleri sonucunda en yüksek potansiyeli gösteren **Random Forest** ve **XGBoost** modelleri seçilerek `GridSearchCV` yöntemi ile derinlemesine hiperparametre ayarlaması (tuning) yapılmıştır. Model karmaşıklığını kontrol altında tutarak maksimum genelleme yeteneğine ulaşılması hedeflenmiştir.

## 🚀 Projeyi Çalıştırma

Projeyi kendi ortamınızda test etmek için terminal veya komut satırında aşağıdaki adımları izleyebilirsiniz:

1. Depoyu bilgisayarınıza klonlayın.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost lightgbm
