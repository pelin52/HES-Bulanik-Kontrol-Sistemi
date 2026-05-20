import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg') # Grafik kilitlenmesini önleyen kritik satır
import matplotlib.pyplot as plt

# =====================================================================
# BARAJ KONTROL PROJESİ - BULANIK MANTIK MOTORU
# =====================================================================

class EndustriyelBarajKontrolMotoru:
    def __init__(self):
        self.evren_doluluk = np.arange(0, 101, 1)     
        self.evren_debi    = np.arange(0, 1001, 1)   
        self.evren_fiyat   = np.arange(0, 5001, 1)   
        self.evren_kontrol = np.arange(0, 101, 1)     
        
        self.bulanik_mimarini_kur()

    def bulanik_mimarini_kur(self):
        # Üyelik Fonksiyonları (Ölü nokta kalmayacak şekilde genişletilmiş endüstriyel aralıklar)
        self.mf_doluluk_kritik = fuzz.trapmf(self.evren_doluluk, [0, 0, 35, 55])
        self.mf_doluluk_nominal = fuzz.trimf(self.evren_doluluk, [35, 55, 75])
        self.mf_doluluk_tasma   = fuzz.trapmf(self.evren_doluluk, [55, 75, 100, 100])

        self.mf_debi_kurak  = fuzz.trapmf(self.evren_debi, [0, 0, 250, 450])
        self.mf_debi_normal = fuzz.trimf(self.evren_debi, [250, 500, 750])
        self.mf_debi_taskin = fuzz.trapmf(self.evren_debi, [550, 750, 1000, 1000])

        self.mf_fiyat_ucuz    = fuzz.trapmf(self.evren_fiyat, [0, 0, 1500, 2500])
        self.mf_fiyat_standart = fuzz.trimf(self.evren_fiyat, [1500, 2700, 3800])
        self.mf_fiyat_pahali   = fuzz.trapmf(self.evren_fiyat, [2800, 4000, 5000, 5000])

        self.mf_kontrol_eko     = fuzz.trimf(self.evren_kontrol, [0, 0, 45])
        self.mf_kontrol_dengeli = fuzz.trimf(self.evren_kontrol, [30, 50, 70])
        self.mf_kontrol_tahliye = fuzz.trapmf(self.evren_kontrol, [55, 75, 100, 100])

    def kurallari_degerlendir(self, v1, v2, v3):
        # Üyelik derecelerini hesapla
        mu_d_k = fuzz.interp_membership(self.evren_doluluk, self.mf_doluluk_kritik, v1)
        mu_d_n = fuzz.interp_membership(self.evren_doluluk, self.mf_doluluk_nominal, v1)
        mu_d_t = fuzz.interp_membership(self.evren_doluluk, self.mf_doluluk_tasma, v1)

        mu_deb_k = fuzz.interp_membership(self.evren_debi, self.mf_debi_kurak, v2)
        mu_deb_n = fuzz.interp_membership(self.evren_debi, self.mf_debi_normal, v2)
        mu_deb_t = fuzz.interp_membership(self.evren_debi, self.mf_debi_taskin, v2)

        mu_f_u = fuzz.interp_membership(self.evren_fiyat, self.mf_fiyat_ucuz, v3)
        mu_f_s = fuzz.interp_membership(self.evren_fiyat, self.mf_fiyat_standart, v3)
        mu_f_p = fuzz.interp_membership(self.evren_fiyat, self.mf_fiyat_pahali, v3)

        # 15 Kuralın Aktivasyon Gücü (Mamdani T-Norm: MIN)
        kural_gcleri = [
            min(mu_d_t, mu_deb_t),   # K1
            min(mu_d_t, mu_deb_n),   # K2
            min(mu_d_n, mu_deb_t),   # K3
            min(mu_d_k, mu_deb_t),   # K4
            min(mu_d_t, mu_f_p),     # K5
            min(mu_d_n, mu_f_p),     # K6
            min(mu_d_n, mu_f_s),     # K7
            min(mu_d_k, mu_f_p),     # K8
            min(mu_d_k, mu_f_u),     # K9
            min(mu_d_k, mu_f_s),     # K10
            min(mu_d_n, mu_f_u),     # K11
            min(mu_deb_k, mu_f_u),   # K12
            min(mu_d_n, mu_deb_n),   # K13
            min(mu_deb_n, mu_f_s),   # K14
            min(mu_d_t, mu_f_u)      # K15
        ]

        # Mamdani Çıkış Alanı Birleştirme (MAX Operatörü)
        eko_alan     = np.zeros_like(self.evren_kontrol)
        dengeli_alan = np.zeros_like(self.evren_kontrol)
        tahliye_alan = np.zeros_like(self.evren_kontrol)

        # Kuralların çıkış kümelerini kesmesi
        guc_eko = max(kural_gcleri[8], kural_gcleri[9], kural_gcleri[10], kural_gcleri[11])
        eko_alan = np.fmin(guc_eko, self.mf_kontrol_eko)

        guc_dengeli = max(kural_gcleri[3], kural_gcleri[6], kural_gcleri[7], kural_gcleri[12], kural_gcleri[13], kural_gcleri[14])
        dengeli_alan = np.fmin(guc_dengeli, self.mf_kontrol_dengeli)

        guc_tahliye = max(kural_gcleri[0], kural_gcleri[1], kural_gcleri[2], kural_gcleri[4], kural_gcleri[5])
        tahliye_alan = np.fmin(guc_tahliye, self.mf_kontrol_tahliye)

        # Toplam Birleşik Çıkış Alanı
        birlesik_alan = np.fmax(eko_alan, np.fmax(dengeli_alan, tahliye_alan))

        # Durulaştırma (Centroid)
        try:
            if np.sum(birlesik_alan) > 0:
                cikis_v = fuzz.defuzz(self.evren_kontrol, birlesik_alan, 'centroid')
            else:
                cikis_v = 50.0
        except:
            cikis_v = 50.0

        return cikis_v, kural_gcleri, birlesik_alan

@st.cache_data(show_spinner=False)
def hesapla_3d_baraj_yuzey_hizli(val_fiyat):
    motor_gecici = EndustriyelBarajKontrolMotoru()
    x_mesh, y_mesh = np.meshgrid(np.linspace(0, 100, 20), np.linspace(0, 1000, 20))
    z_mesh = np.zeros_like(x_mesh)
    
    for i in range(20):
        for j in range(20):
            cikis, _, _ = motor_gecici.kurallari_degerlendir(x_mesh[i, j], y_mesh[i, j], val_fiyat)
            z_mesh[i, j] = cikis
    return x_mesh, y_mesh, z_mesh

# =====================================================================
# STREAMLIT KULLANICI ARAYÜZÜ (GUI)
# =====================================================================
st.set_page_config(page_title="Baraj Bulanık Mantık Kontrolü", layout="wide")
st.title("Bulanık Mantık ile HES Baraj Kontrol ve Otomasyon Sistemi Projesi")
st.write("Hazırlayan: Pelin - Bulanık Mantık Dönem Projesi Analiz Ekranı")
st.markdown("---")

if 'baraj_motor' not in st.session_state:
    st.session_state.baraj_motor = EndustriyelBarajKontrolMotoru()
motor = st.session_state.baraj_motor

# Yan Panel Girdileri
st.sidebar.header("Sensör Ayarları (Girişler)")
val_doluluk = st.sidebar.slider("Baraj Doluluk Oranı (%)", 0, 100, 75)
val_debi    = st.sidebar.slider("Gelen Nehir Debisi (m³/s)", 0, 1000, 450)
val_fiyat   = st.sidebar.slider("Spot Elektrik Fiyatı (TL)", 0, 5000, 3200)

# Hesaplamayı sekme dışına, en başta yapıyoruz ki değerler kaybolmasın
cikis_v, kural_gcleri, birlesik_alan = motor.kurallari_degerlendir(val_doluluk, val_debi, val_fiyat)

# Üst Metrik Paneli
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(label="Girilen Değerler [Doluluk | Debi | Fiyat]", value=f"%{val_doluluk} -- {val_debi} m³/s -- {val_fiyat} TL")
with k2:
    st.metric(label="Çıkış Değeri (Sistem Aktivasyonu Y*)", value=f"%{cikis_v:.2f}")
with k3:
    if cikis_v <= 40:
        mod_isimlendirme = "Ekonomik Koruma Modu (Su Tutuluyor)"
    elif 40 < cikis_v <= 70:
        mod_isimlendirme = "Dengeli Üretim Modu (Normal Çalışma)"
    else:
        mod_isimlendirme = "Maksimum Tahliye Modu (Acil Durum Kapaklar Açık)"
    st.metric(label="Sistem Çalışma Modu", value=mod_isimlendirme)

st.markdown("---")

# Kural tanımlarını bir liste haline getirdik (Döngüde kaybolmaması için garanti yöntem)
tanimlar = [
    "EĞER Doluluk = Taşma Sınırı VE Debi = Taşkın İSE Sistem = Maksimum Tahliye",
    "EĞER Doluluk = Taşma Sınırı VE Debi = Normal İSE Sistem = Maksimum Tahliye",
    "EĞER Doluluk = Nominal VE Debi = Taşkın İSE Sistem = Maksimum Tahliye",
    "EĞER Doluluk = Kritik Düşük VE Debi = Taşkın İSE Sistem = Dengeli Üretim",
    "EĞER Doluluk = Taşma Sınırı VE Fiyat = Pahalı İSE Sistem = Maksimum Tahliye",
    "EĞER Doluluk = Nominal VE Fiyat = Pahalı İSE Sistem = Maksimum Tahliye",
    "EĞER Doluluk = Nominal VE Fiyat = Standart İSE Sistem = Dengeli Üretim",
    "EĞER Doluluk = Kritik Düşük VE Fiyat = Pahalı İSE Sistem = Dengeli Üretim",
    "EĞER Doluluk = Kritik Düşük VE Fiyat = Ucuz İSE Sistem = Ekonomik Koruma",
    "EĞER Doluluk = Kritik Düşük VE Fiyat = Standart İSE Sistem = Ekonomik Koruma",
    "EĞER Doluluk = Nominal VE Fiyat = Ucuz İSE Sistem = Ekonomik Koruma",
    "EĞER Debi = Kurak VE Fiyat = Ucuz İSE Sistem = Ekonomik Koruma",
    "EĞER Doluluk = Nominal VE Debi = Normal İSE Sistem = Dengeli Üretim",
    "EĞER Debi = Normal VE Fiyat = Standart İSE Sistem = Dengeli Üretim",
    "EĞER Doluluk = Taşma Sınırı VE Fiyat = Ucuz İSE Sistem = Dengeli Üretim"
]

# Ana ekran düzeni için sekmeler (Grafikler buraya gelecek)
sekme_uf, sekme_inference = st.tabs(["1. Giriş Değerleri ve Üyelik Grafikleri", "2. Kural Karar Mekanizması"])

with sekme_uf:
    st.write("### Girişlerin Grafikte Gösterimi ve Üyelik Dereceleri")
    
    mu_d_k = fuzz.interp_membership(motor.evren_doluluk, motor.mf_doluluk_kritik, val_doluluk)
    mu_d_n = fuzz.interp_membership(motor.evren_doluluk, motor.mf_doluluk_nominal, val_doluluk)
    mu_d_t = fuzz.interp_membership(motor.evren_doluluk, motor.mf_doluluk_tasma, val_doluluk)

    mu_deb_k = fuzz.interp_membership(motor.evren_debi, motor.mf_debi_kurak, val_debi)
    mu_deb_n = fuzz.interp_membership(motor.evren_debi, motor.mf_debi_normal, val_debi)
    mu_deb_t = fuzz.interp_membership(motor.evren_debi, motor.mf_debi_taskin, val_debi)

    mu_f_u = fuzz.interp_membership(motor.evren_fiyat, motor.mf_fiyat_ucuz, val_fiyat)
    mu_f_s = fuzz.interp_membership(motor.evren_fiyat, motor.mf_fiyat_standart, val_fiyat)
    mu_f_p = fuzz.interp_membership(motor.evren_fiyat, motor.mf_fiyat_pahali, val_fiyat)

    g1, g2, g3 = st.columns(3)
    
    with g1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(motor.evren_doluluk, motor.mf_doluluk_kritik, 'b', label='Kritik Düşük')
        ax1.plot(motor.evren_doluluk, motor.mf_doluluk_nominal, 'g', label='Nominal')
        ax1.plot(motor.evren_doluluk, motor.mf_doluluk_tasma, 'r', label='Taşma Sınırı')
        ax1.axvline(x=val_doluluk, color='black', linestyle='--')
        ax1.set_title(f"Giriş 1: Baraj Doluluğu\nKritik:{mu_d_k:.1f} | Nom:{mu_d_n:.1f} | Taşma:{mu_d_t:.1f}", fontsize=8)
        ax1.legend(fontsize=6)
        st.pyplot(fig1)
        plt.close(fig1)

    with g2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(motor.evren_debi, motor.mf_debi_kurak, 'b', label='Kurak')
        ax2.plot(motor.evren_debi, motor.mf_debi_normal, 'g', label='Normal')
        ax2.plot(motor.evren_debi, motor.mf_debi_taskin, 'r', label='Taşkın')
        ax2.axvline(x=val_debi, color='black', linestyle='--')
        ax2.set_title(f"Giriş 2: Gelen Debi\nKurak:{mu_deb_k:.1f} | Norm:{mu_deb_n:.1f} | Taşkın:{mu_deb_t:.1f}", fontsize=8)
        ax2.legend(fontsize=6)
        st.pyplot(fig2)
        plt.close(fig2)

    with g3:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.plot(motor.evren_fiyat, motor.mf_fiyat_ucuz, 'b', label='Ucuz')
        ax3.plot(motor.evren_fiyat, motor.mf_fiyat_standart, 'g', label='Standart')
        ax3.plot(motor.evren_fiyat, motor.mf_fiyat_pahali, 'r', label='Pahalı')
        ax3.axvline(x=val_fiyat, color='black', linestyle='--')
        ax3.set_title(f"Giriş 3: Elektrik Fiyatı\nUcuz:{mu_f_u:.1f} | Standart:{mu_f_s:.1f} | Pahalı:{mu_f_p:.1f}", fontsize=8)
        ax3.legend(fontsize=6)
        st.pyplot(fig3)
        plt.close(fig3)

with sekme_inference:
    st.write("### Kuralların Aktivasyon Dereceleri ve Çıkış Alanı Grafikleri")
    col_inf1, col_inf2 = st.columns(2)
    
    with col_inf1:
        st.write("Grafik: 15 Kuralın Tamamının Aktivasyon Dereceleri")
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        x_etiketler = [f"K{i+1}" for i in range(15)]
        ax4.bar(x_etiketler, kural_gcleri, color='royalblue', edgecolor='black')
        ax4.set_ylabel("Aktivasyon Derecesi")
        ax4.set_ylim(0, 1.1)
        ax4.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig4)
        plt.close(fig4)

    with col_inf2:
        st.write("Grafik: Birleşik Çıkış Alanı ve Çıkış Noktası")
        fig5, ax5 = plt.subplots(figsize=(5, 3.5))
        ax5.plot(motor.evren_kontrol, motor.mf_kontrol_eko, 'b--', label='Eko Koruma')
        ax5.plot(motor.evren_kontrol, motor.mf_kontrol_dengeli, 'g--', label='Dengeli Ur.')
        ax5.plot(motor.evren_kontrol, motor.mf_kontrol_tahliye, 'r--', label='Mak. Tahliye')
        ax5.fill_between(motor.evren_kontrol, 0, birlesik_alan, facecolor='orange', alpha=0.5, label='Birleşik Çıktı Alanı')
        ax5.axvline(x=cikis_v, color='red', linewidth=2, label=f'Hesaplanan Nokta (%{cikis_v:.2f})')
        ax5.legend(fontsize=7)
        st.pyplot(fig5)
        plt.close(fig5)

# =====================================================================
# GARANTİLİ KURAL LİSTELEME ALANI (SEKME DIŞINA TAŞINDI)
# =====================================================================
st.markdown("---")
st.subheader("🔥 Canlı Aktif Kural Kontrol ve Endüstriyel Hata Ayıklama Paneli")
st.write("Mevcut sensör girdilerine göre tetiklenen kuralların güncel durumu aşağıdadır:")

# Yazıların sekmelerin azizliğine uğramaması için doğrudan ana ekrana döngü ile basıyoruz
for idx, guc in enumerate(kural_gcleri):
    if guc > 0:
        st.success(f"**[Kural {idx+1} Aktif]** (Tetiklenme Gücü: `{guc:.2f}`) ➔ {tanimlar[idx]}")
    else:
        st.text(f"[Kural {idx+1} Pasif] (Tetiklenme Gücü: 0.00) ➔ {tanimlar[idx]}")

# 3D Karar Yüzeyi
st.markdown("---")
st.write("### Sistemin 3D Çıktı Karar Yüzeyi")

x_mesh, y_mesh, z_mesh = hesapla_3d_baraj_yuzey_hizli(val_fiyat)
fig6 = plt.figure(figsize=(8, 4))
ax6 = fig6.add_subplot(111, projection='3d')
surf = ax6.plot_surface(x_mesh, y_mesh, z_mesh, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax6.set_xlabel('Baraj Dolulugu (%)', fontsize=8)
ax6.set_ylabel('Gelen Debi (m3/s)', fontsize=8)
ax6.set_zlabel('Sistem Aktivasyonu (%)', fontsize=8)
st.pyplot(fig6)
plt.close(fig6)