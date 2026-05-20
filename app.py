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
        self.evren_debi    = np.arange(0, 1001, 10)   
        self.evren_fiyat   = np.arange(0, 5001, 50)   
        self.evren_kontrol = np.arange(0, 101, 1)     
        
        self.bulanik_mimarini_kur()

    def bulanik_mimarini_kur(self):
        self.doluluk = ctrl.Antecedent(self.evren_doluluk, 'Baraj_Doluluk')
        self.debi    = ctrl.Antecedent(self.evren_debi, 'Gelen_Debi')
        self.fiyat   = ctrl.Antecedent(self.evren_fiyat, 'Elektrik_Fiyati')
        self.kontrol = ctrl.Consequent(self.evren_kontrol, 'Sistem_Aktivasyon')

        # Üyelik Fonksiyonları
        self.doluluk['Kritik_Dusuk'] = fuzz.trapmf(self.evren_doluluk, [0, 0, 20, 45])
        self.doluluk['Nominal']      = fuzz.trimf(self.evren_doluluk, [30, 55, 80])
        self.doluluk['Tasma_Siniri'] = fuzz.trapmf(self.evren_doluluk, [65, 85, 100, 100])

        self.debi['Kurak']  = fuzz.trapmf(self.evren_debi, [0, 0, 200, 450])
        self.debi['Normal'] = fuzz.trimf(self.evren_debi, [300, 500, 750])
        self.debi['Taskin'] = fuzz.trapmf(self.evren_debi, [600, 800, 1000, 1000])

        self.fiyat['Ucuz']    = fuzz.trapmf(self.evren_fiyat, [0, 0, 1000, 2200])
        self.fiyat['Standart'] = fuzz.trimf(self.evren_fiyat, [1500, 2700, 3800])
        self.fiyat['Pahali']   = fuzz.trapmf(self.evren_fiyat, [3000, 4200, 5000, 5000])

        self.kontrol['Ekonomik_Koruma'] = fuzz.trimf(self.evren_kontrol, [0, 0, 40])
        self.kontrol['Dengeli_Uretim']   = fuzz.trimf(self.evren_kontrol, [30, 50, 70])
        self.kontrol['Maksimum_Tahliye'] = fuzz.trapmf(self.evren_kontrol, [60, 80, 100, 100])

        # Kural Tabanı
        self.kurallar = [
            ctrl.Rule(self.doluluk['Tasma_Siniri'] & self.debi['Taskin'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Tasma_Siniri'] & self.debi['Normal'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Nominal'] & self.debi['Taskin'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.debi['Taskin'], self.kontrol['Dengeli_Uretim']),
            
            ctrl.Rule(self.doluluk['Tasma_Siniri'] & self.fiyat['Pahali'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Nominal'] & self.fiyat['Pahali'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Nominal'] & self.fiyat['Standart'], self.kontrol['Dengeli_Uretim']),
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.fiyat['Pahali'], self.kontrol['Dengeli_Uretim']),
            
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.fiyat['Ucuz'], self.kontrol['Ekonomik_Koruma']),
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.fiyat['Standart'], self.kontrol['Ekonomik_Koruma']),
            ctrl.Rule(self.doluluk['Nominal'] & self.fiyat['Ucuz'], self.kontrol['Ekonomik_Koruma']),
            ctrl.Rule(self.debi['Kurak'] & self.fiyat['Ucuz'], self.kontrol['Ekonomik_Koruma']),
            
            ctrl.Rule(self.doluluk['Nominal'] & self.debi['Normal'], self.kontrol['Dengeli_Uretim']),
            ctrl.Rule(self.debi['Normal'] & self.fiyat['Standart'], self.kontrol['Dengeli_Uretim']),
            ctrl.Rule(self.doluluk['Tasma_Siniri'] & self.fiyat['Ucuz'], self.kontrol['Dengeli_Uretim'])
        ]
        
        self.kontrol_sistemi = ctrl.ControlSystem(self.kurallar)
        self.simulasyon = ctrl.ControlSystemSimulation(self.kontrol_sistemi)

    def hesapla(self, v1, v2, v3, defuzz_method):
        try:
            self.kontrol.defuzzify_method = defuzz_method
            self.simulasyon.input['Baraj_Doluluk'] = v1
            self.simulasyon.input['Gelen_Debi'] = v2
            self.simulasyon.input['Elektrik_Fiyati'] = v3
            self.simulasyon.compute()
            return self.simulasyon.output['Sistem_Aktivasyon'], self.simulasyon
        except Exception as e:
            return 50.0, self.simulasyon

@st.cache_data(show_spinner=False)
def hesapla_3d_baraj_yuzey(val_fiyat):
    motor_gecici = EndustriyelBarajKontrolMotoru()
    x_mesh, y_mesh = np.meshgrid(np.linspace(0, 100, 15), np.linspace(0, 1000, 15))
    z_mesh = np.zeros_like(x_mesh)
    yuz_sim = ctrl.ControlSystemSimulation(motor_gecici.kontrol_sistemi)
    yuz_sim.input['Elektrik_Fiyati'] = val_fiyat
    
    for i in range(15):
        for j in range(15):
            yuz_sim.input['Baraj_Doluluk'] = x_mesh[i, j]
            yuz_sim.input['Gelen_Debi'] = y_mesh[i, j]
            try:
                yuz_sim.compute()
                z_mesh[i, j] = yuz_sim.output['Sistem_Aktivasyon']
            except:
                z_mesh[i, j] = 50.0
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

# Yan Panel
st.sidebar.header("Sensör Ayarları (Girişler)")
val_doluluk = st.sidebar.slider("Baraj Doluluk Oranı (%)", 0, 100, 75)
val_debi    = st.sidebar.slider("Gelen Nehir Debisi (m3/s)", 0, 1000, 450)
val_fiyat   = st.sidebar.slider("Spot Elektrik Fiyatı (TL)", 0, 5000, 3200)

st.sidebar.markdown("---")
metot = st.sidebar.selectbox(
    "Durulaştırma Yöntemi (Defuzzification)",
    ["centroid", "bisector", "mom"],
    format_func=lambda x: "Ağırlık Merkezi (Centroid)" if x=="centroid" else "Alana Bölen (Bisector)" if x=="bisector" else "Maksimumlar Ortalaması (MOM)"
)

cikis_v, sim_aktuel = motor.hesapla(val_doluluk, val_debi, val_fiyat, metot)

# Metrikler
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(label="Girilen Değerler [Doluluk | Debi | Fiyat]", value=f"%{val_doluluk} -- {val_debi} m³s -- {val_fiyat} TL")
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

sekme_uf, sekme_inference = st.tabs(["1. Giriş Değerleri ve Üyelik Grafikleri", "2. Kural Karar Mekanizması"])

with sekme_uf:
    st.write("### Girişlerin Grafikte Gösterimi ve Üyelik Dereceleri")
    
    mu_d_k = fuzz.interp_membership(motor.evren_doluluk, motor.doluluk['Kritik_Dusuk'].mf, val_doluluk)
    mu_d_n = fuzz.interp_membership(motor.evren_doluluk, motor.doluluk['Nominal'].mf, val_doluluk)
    mu_d_t = fuzz.interp_membership(motor.evren_doluluk, motor.doluluk['Tasma_Siniri'].mf, val_doluluk)

    mu_deb_k = fuzz.interp_membership(motor.evren_debi, motor.debi['Kurak'].mf, val_debi)
    mu_deb_n = fuzz.interp_membership(motor.evren_debi, motor.debi['Normal'].mf, val_debi)
    mu_deb_t = fuzz.interp_membership(motor.evren_debi, motor.debi['Taskin'].mf, val_debi)

    mu_f_u = fuzz.interp_membership(motor.evren_fiyat, motor.fiyat['Ucuz'].mf, val_fiyat)
    mu_f_s = fuzz.interp_membership(motor.evren_fiyat, motor.fiyat['Standart'].mf, val_fiyat)
    mu_f_p = fuzz.interp_membership(motor.evren_fiyat, motor.fiyat['Pahali'].mf, val_fiyat)

    g1, g2, g3 = st.columns(3)
    
    with g1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(motor.evren_doluluk, motor.doluluk['Kritik_Dusuk'].mf, 'b', label='Kritik Düşük')
        ax1.plot(motor.evren_doluluk, motor.doluluk['Nominal'].mf, 'g', label='Nominal')
        ax1.plot(motor.evren_doluluk, motor.doluluk['Tasma_Siniri'].mf, 'r', label='Taşma Sınırı')
        ax1.axvline(x=val_doluluk, color='black', linestyle='--')
        ax1.set_title(f"Giriş 1: Baraj Doluluğu\nKritik:{mu_d_k:.1f} | Nom:{mu_d_n:.1f} | Taşma:{mu_d_t:.1f}", fontsize=8)
        ax1.legend(fontsize=6)
        st.pyplot(fig1)
        plt.close(fig1)

    with g2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(motor.evren_debi, motor.debi['Kurak'].mf, 'b', label='Kurak')
        ax2.plot(motor.evren_debi, motor.debi['Normal'].mf, 'g', label='Normal')
        ax2.plot(motor.evren_debi, motor.debi['Taskin'].mf, 'r', label='Taşkın')
        ax2.axvline(x=val_debi, color='black', linestyle='--')
        ax2.set_title(f"Giriş 2: Gelen Debi\nKurak:{mu_deb_k:.1f} | Norm:{mu_deb_n:.1f} | Taşkın:{mu_deb_t:.1f}", fontsize=8)
        ax2.legend(fontsize=6)
        st.pyplot(fig2)
        plt.close(fig2)

    with g3:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.plot(motor.evren_fiyat, motor.fiyat['Ucuz'].mf, 'b', label='Ucuz')
        ax3.plot(motor.evren_fiyat, motor.fiyat['Standart'].mf, 'g', label='Standart')
        ax3.plot(motor.evren_fiyat, motor.fiyat['Pahali'].mf, 'r', label='Pahalı')
        ax3.axvline(x=val_fiyat, color='black', linestyle='--')
        ax3.set_title(f"Giriş 3: Elektrik Fiyatı\nUcuz:{mu_f_u:.1f} | Standart:{mu_f_s:.1f} | Pahalı:{mu_f_p:.1f}", fontsize=8)
        ax3.legend(fontsize=6)
        st.pyplot(fig3)
        plt.close(fig3)

with sekme_inference:
    st.write("### Kuralların Aktivasyon Değerleri ve Çıkış Alanı Grafikleri")
    col_inf1, col_inf2 = st.columns(2)
    
    kural_gcleri = [
        min(mu_d_t, mu_deb_t), min(mu_d_t, mu_deb_n), min(mu_d_n, mu_deb_t), min(mu_d_k, mu_deb_t),
        min(mu_d_t, mu_f_p), min(mu_d_n, mu_f_p), min(mu_d_n, mu_f_s), min(mu_d_k, mu_f_p),
        min(mu_d_k, mu_f_u), min(mu_d_k, mu_f_s), min(mu_d_n, mu_f_u), min(mu_deb_k, mu_f_u),
        min(mu_d_n, mu_deb_n), min(mu_deb_n, mu_f_s), min(mu_d_t, mu_f_u)
    ]
            
    with col_inf1:
        st.write("Grafik: Kuralların Aktivasyon Dereceleri")
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        x_etiketler = [f"K{i+1}" for i in range(15)]
        ax4.bar(x_etiketler, kural_gcleri, color='blue', edgecolor='black')
        ax4.set_ylabel("Aktivasyon Derecesi")
        ax4.set_ylim(0, 1.1)
        st.pyplot(fig4)
        plt.close(fig4)

    with col_inf2:
        st.write("Grafik: Birleşik Çıkış Alanı ve Çıkış Noktası")
        fig5, ax5 = plt.subplots(figsize=(5, 3.5))
        ax5.plot(motor.evren_kontrol, motor.kontrol['Ekonomik_Koruma'].mf, 'b--', label='Eko Koruma')
        ax5.plot(motor.evren_kontrol, motor.kontrol['Dengeli_Uretim'].mf, 'g--', label='Dengeli Ur.')
        ax5.plot(motor.evren_kontrol, motor.kontrol['Maksimum_Tahliye'].mf, 'r--', label='Mak. Tahliye')
        
        if sim_aktuel:
            try:
                gercek_alan = motor.kontrol.testing_membership[sim_aktuel]
                ax5.fill_between(motor.evren_kontrol, 0, gercek_alan, facecolor='orange', alpha=0.5, label='Çıktı Alanı')
            except:
                pass
        
        ax5.axvline(x=cikis_v, color='red', linewidth=2, label=f'Hesaplanan Nokta (%{cikis_v:.2f})')
        ax5.legend(fontsize=7)
        st.pyplot(fig5)
        plt.close(fig5)

    # Canlı kural Kontrol Paneli
    st.markdown("---")
    st.write("### Canlı Aktif Kural Kontrolü (Hata Ayıklama Paneli)")
    st.write("Mevcut girdilere göre karara etki eden aktif kurallar:")
    
    if kural_gcleri[0] > 0: st.info(f"-> **[Kural 1 Aktif (Güç: {kural_gcleri[0]:.2f})]** EĞER Doluluk = Taşma Sınırı VE Debi = Taşkın İSE Sistem = Maksimum Tahliye")
    if kural_gcleri[1] > 0: st.info(f"-> **[Kural 2 Aktif (Güç: {kural_gcleri[1]:.2f})]** EĞER Doluluk = Taşma Sınırı VE Debi = Normal İSE Sistem = Maksimum Tahliye")
    if kural_gcleri[2] > 0: st.info(f"-> **[Kural 3 Aktif (Güç: {kural_gcleri[2]:.2f})]** EĞER Doluluk = Nominal VE Debi = Taşkın İSE Sistem = Maksimum Tahliye")
    if kural_gcleri[3] > 0: st.info(f"-> **[Kural 4 Aktif (Güç: {kural_gcleri[3]:.2f})]** EĞER Doluluk = Kritik Düşük VE Debi = Taşkın İSE Sistem = Dengeli Üretim")
    if kural_gcleri[4] > 0: st.info(f"-> **[Kural 5 Aktif (Güç: {kural_gcleri[4]:.2f})]** EĞER Doluluk = Taşma Sınırı VE Fiyat = Pahalı İSE Sistem = Maksimum Tahliye")
    if kural_gcleri[5] > 0: st.info(f"-> **[Kural 6 Aktif (Güç: {kural_gcleri[5]:.2f})]** EĞER Doluluk = Nominal VE Fiyat = Pahalı İSE Sistem = Maksimum Tahliye")
    if kural_gcleri[6] > 0: st.info(f"-> **[Kural 7 Aktif (Güç: {kural_gcleri[6]:.2f})]** EĞER Doluluk = Nominal VE Fiyat = Standart İSE Sistem = Dengeli Üretim")
    if kural_gcleri[7] > 0: st.info(f"-> **[Kural 8 Aktif (Güç: {kural_gcleri[7]:.2f})]** EĞER Doluluk = Kritik Düşük VE Fiyat = Pahalı İSE Sistem = Dengeli Üretim")
    if kural_gcleri[8] > 0: st.info(f"-> **[Kural 9 Aktif (Güç: {kural_gcleri[8]:.2f})]** EĞER Doluluk = Kritik Düşük VE Fiyat = Ucuz İSE Sistem = Ekonomik Koruma")
    if kural_gcleri[9] > 0: st.info(f"-> **[Kural 10 Aktif (Güç: {kural_gcleri[9]:.2f})]** EĞER Doluluk = Kritik Düşük VE Fiyat = Standart İSE Sistem = Ekonomik Koruma")
    if kural_gcleri[10] > 0: st.info(f"-> **[Kural 11 Aktif (Güç: {kural_gcleri[10]:.2f})]** EĞER Doluluk = Nominal VE Fiyat = Ucuz İSE Sistem = Ekonomik Koruma")
    if kural_gcleri[11] > 0: st.info(f"-> **[Kural 12 Aktif (Güç: {kural_gcleri[11]:.2f})]** EĞER Debi = Kurak VE Fiyat = Ucuz İSE Sistem = Ekonomik Koruma")
    if kural_gcleri[12] > 0: st.info(f"-> **[Kural 13 Aktif (Güç: {kural_gcleri[12]:.2f})]** EĞER Doluluk = Nominal VE Debi = Normal İSE Sistem = Dengeli Üretim")
    if kural_gcleri[13] > 0: st.info(f"-> **[Kural 14 Aktif (Güç: {kural_gcleri[13]:.2f})]** EĞER Debi = Normal VE Fiyat = Standart İSE Sistem = Dengeli Üretim")
    if kural_gcleri[14] > 0: st.info(f"-> **[Kural 15 Aktif (Güç: {kural_gcleri[14]:.2f})]** EĞER Doluluk = Taşma Sınırı VE Fiyat = Ucuz İSE Sistem = Dengeli Üretim")

    # 3D Karar Yüzeyi
    st.markdown("---")
    st.write("### Sistemin 3D Çıktı Karar Yüzeyi")
    
    x_mesh, y_mesh, z_mesh = hesapla_3d_baraj_yuzey(val_fiyat)
    fig6 = plt.figure(figsize=(8, 4))
    ax6 = fig6.add_subplot(111, projection='3d')
    surf = ax6.plot_surface(x_mesh, y_mesh, z_mesh, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax6.set_xlabel('Baraj Dolulugu (%)', fontsize=8)
    ax6.set_ylabel('Gelen Debi (m3/s)', fontsize=8)
    ax6.set_zlabel('Sistem Aktivasyonu (%)', fontsize=8)
    st.pyplot(fig6)
    plt.close(fig6)