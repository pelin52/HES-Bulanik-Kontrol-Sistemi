# HES Bulanik Kontrol Sistemi
HES Bulanık Kontrol Sistemi, barajlardaki doluluk oranını, nehirlerden gelen su miktarını (debi) ve elektrik piyasasındaki anlık fiyatları inceleyerek; barajın hem güvenli kalmasını hem de en yüksek kârla elektrik üretmesini sağlayan bulanık mantık mekanizmasıdır.
## Kullanılan Kütüphaneler
`tkinter`
`numpy`
`scikit-fuzzy`
`matplotlib`
## 🌟 Öne Çıkan Özellikler

* **🧠 Mamdani Bulanık Çıkarım Motoru:** * 3 adet dinamik giriş (**Doluluk**, **Debi**, **Fiyat**) ve 1 adet mekanik çıkış (**Valf Kontrolü**) değişkeni.
  * Mantıksal çelişki içermeyen, risk ve kazanç dengesini optimize eden **15 kurallı** gelişmiş karar mimarisi.

* **🔄 Canlı SCADA Simülasyonu:** * Gerçek dünya borsa hareketlerini (gece-gündüz fiyat dalgalanmaları) ve anlık nehir debisi değişimlerini simüle eden reaktif ve dinamik mod.

* **📊 Gerçek Zamanlı Hidrolik Güç ve Finansal Model:** * Bulanık çıkarımdan elde edilen anlık valf açıklığına göre üretilen aktif gücü ($MW$) hesaplayan hidrolik denklem katmanı.
  * Üretilen enerji ve anlık piyasa fiyatı üzerinden kazanılan toplam finansal ciroyu ($TL$) saniyelik olarak güncelleyen finansal takip motoru.

* **🚨 SCADA Erken Uyarı ve Alarm Yönetimi:** * **Taşkın Tehlikesi:** Yüksek doluluk ve aşırı debi durumlarında devreye giren kritik güvenlik koruması.
  * **Kavitasyon Riski:** Düşük su seviyesinde türbinlerin zarar görmesini engellemek için operatörü görsel olarak uyaran mekanik koruma algoritması.

* **📈 Gelişmiş Görselleştirme (Data Visualization):** * Sentezlenmiş bulanık karar alanını ve **Ağırlık Merkezini (Centroid)** arayüze entegre şekilde anlık çizen grafik ekranı.
  * Giriş parametrelerine ait Üyelik Fonksiyonlarını (Yamuk ve Üçgen MF) gösteren özel analiz sekmesi (Matplotlib & Tkinter entegrasyonu).
