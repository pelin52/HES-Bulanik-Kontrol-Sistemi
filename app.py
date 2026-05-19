import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# =====================================================================
# 1. ENDÜSTRİYEL BULANIK MANTIK MOTORU (MAMDANI HESAPLAMA YAPISI)
# =====================================================================
class EndustriyelBarajKontrolMotoru:
    def __init__(self):
        # Gerçek Dünya Fiziksel Evrenleri
        self.evren_doluluk = np.arange(0, 101, 1)     # Baraj Doluluk Oranı (%)
        self.evren_debi    = np.arange(0, 1001, 10)   # Nehir Gelen Su Debisi (m³/s)
        self.evren_fiyat   = np.arange(0, 5001, 50)   # Elektrik Spot Piyasa Fiyatı (TL/MWh)
        self.evren_kontrol = np.arange(0, 101, 1)     # Çıkış: Tahliye & Üretim Modu (%)
        
        self.bulanik_mimarini_kur()

    def bulanik_mimarini_kur(self):
        # Antecedents & Consequent Tanımlamaları
        self.doluluk = ctrl.Antecedent(self.evren_doluluk, 'Baraj_Doluluk')
        self.debi    = ctrl.Antecedent(self.evren_debi, 'Gelen_Debi')
        self.fiyat   = ctrl.Antecedent(self.evren_fiyat, 'Elektrik_Fiyati')
        self.kontrol = ctrl.Consequent(self.evren_kontrol, 'Sistem_Aktivasyon')

        # --- 1. ADIM: BULANIKLAŞTIRMA (FUZZIFICATION) ---
        self.doluluk['Kritik_Dusuk'] = fuzz.trapmf(self.evren_doluluk, [0, 0, 20, 45])
        self.doluluk['Nominal']      = fuzz.trimf(self.evren_doluluk, [30, 55, 80])
        self.doluluk['Tasman_Sınırı'] = fuzz.trapmf(self.evren_doluluk, [65, 85, 100, 100])

        self.debi['Kurak']  = fuzz.trapmf(self.evren_debi, [0, 0, 200, 450])
        self.debi['Normal'] = fuzz.trimf(self.evren_debi, [300, 500, 750])
        self.debi['Taskın'] = fuzz.trapmf(self.evren_debi, [600, 800, 1000, 1000])

        self.fiyat['Ucuz']    = fuzz.trapmf(self.evren_fiyat, [0, 0, 1000, 2200])
        self.fiyat['Standart'] = fuzz.trimf(self.evren_fiyat, [1500, 2700, 3800])
        self.fiyat['Pahalı']   = fuzz.trapmf(self.evren_fiyat, [3000, 4200, 5000, 5000])

        # Çıkış Kümesi Kontrol Aralıkları
        self.kontrol['Ekonomik_Koruma'] = fuzz.trimf(self.evren_kontrol, [0, 0, 40])
        self.kontrol['Dengeli_Uretim']   = fuzz.trimf(self.evren_kontrol, [30, 50, 70])
        self.kontrol['Maksimum_Tahliye'] = fuzz.trapmf(self.evren_kontrol, [60, 80, 100, 100])

        # --- 2. ADIM: OPTİMİZE EDİLMİŞ 15 KURAL ---
        self.kurallar = [
            ctrl.Rule(self.doluluk['Tasman_Sınırı'] & self.debi['Taskın'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Tasman_Sınırı'] & self.debi['Normal'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Nominal'] & self.debi['Taskın'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.debi['Taskın'], self.kontrol['Dengeli_Uretim']),
            
            ctrl.Rule(self.doluluk['Tasman_Sınırı'] & self.fiyat['Pahalı'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Nominal'] & self.fiyat['Pahalı'], self.kontrol['Maksimum_Tahliye']),
            ctrl.Rule(self.doluluk['Nominal'] & self.fiyat['Standart'], self.kontrol['Dengeli_Uretim']),
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.fiyat['Pahalı'], self.kontrol['Dengeli_Uretim']),
            
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.fiyat['Ucuz'], self.kontrol['Ekonomik_Koruma']),
            ctrl.Rule(self.doluluk['Kritik_Dusuk'] & self.fiyat['Standart'], self.kontrol['Ekonomik_Koruma']),
            ctrl.Rule(self.doluluk['Nominal'] & self.fiyat['Ucuz'], self.kontrol['Ekonomik_Koruma']),
            ctrl.Rule(self.debi['Kurak'] & self.fiyat['Ucuz'], self.kontrol['Ekonomik_Koruma']),
            
            ctrl.Rule(self.doluluk['Nominal'] & self.debi['Normal'], self.kontrol['Dengeli_Uretim']),
            ctrl.Rule(self.debi['Normal'] & self.fiyat['Standart'], self.kontrol['Dengeli_Uretim']),
            ctrl.Rule(self.doluluk['Tasman_Sınırı'] & self.fiyat['Ucuz'], self.kontrol['Dengeli_Uretim'])
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
        except Exception:
            return 50.0, self.simulasyon

# 3D Karar Yüzeyi Önbellekleme
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
# 2. KULLANICI ARABİRİMİ TASARIMI (STREAMLIT UI)
# =====================================================================
st.set_page_config(page_title="HES Akıllı Kontrol Sistemi", layout="wide")
st.title("HES Akıllı Otomasyon ve Enerji Optimizasyon Laboratuvarı")
st.caption("Mamdani Bulanık Karar Destek Sistemi Analiz Masası")
st.markdown("---")

if 'baraj_motor' not in st.session_state:
    st.session_state.baraj_motor = EndustriyelBarajKontrolMotoru()
motor = st.session_state.baraj_motor

# --- YAN PANEL ---
st.sidebar.header(" SCADA Canlı Sensör Verileri")
val_doluluk = st.sidebar.slider("Baraj Doluluk Oranı (%)", 0, 100, 75)
val_debi    = st.sidebar.slider("Gelen Nehir Debisi (m³/s)", 0, 1000, 450)
val_fiyat   = st.sidebar.slider("Spot Elektrik Fiyatı (TL/MWh)", 0, 5000, 3200)

st.sidebar.markdown("---")
metot = st.sidebar.selectbox(
    "Durulaştırma Stratejisi",
    ["centroid", "bisector", "mom"],
    format_func=lambda x: "Ağırlık Merkezi (Centroid)" if x=="centroid" else "Alana Bölen (Bisector)" if x=="bisector" else "Maksimumların Ortalaması (MOM)"
)

# Hesaplama
cikis_v, sim_aktuel = motor.hesapla(val_doluluk, val_debi, val_fiyat, metot)

# --- PANEL METRİKLERİ ---
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(label="Anlık Sensör Girdileri [Doluluk | Debi | Fiyat]", value=f"%{val_doluluk} | {val_debi} m³/s | {val_fiyat} TL")
with k2:
    st.metric(label="Hesaplanan Sistem Aktivasyon Gücü (Y*)", value=f"%{cikis_v:.2f}")
with k3:
    if cikis_v <= 40:
        mod_isimlendirme = " Ekonomik Koruma Modu (Su Saklanıyor)"
    elif 40 < cikis_v <= 70:
        mod_isimlendirme = "Dengeli Üretim Modu (Nominal Türbin Aktivasyonu)"
    else:
        mod_isimlendirme = "Maksimum Tahliye Modu (Taşkın ve Risk Yönetimi)"
    st.metric(label="Önerilen SCADA Operasyon Modu", value=mod_isimlendirme)

st.markdown("---")

# --- SEKMELER ---
sekme_uf, sekme_inference = st.tabs([" 1. Aşama: Fuzzification (Sensör Dönüşümü)", " 2-4. Aşama: Mühendislik Karar Mekanizması"])

with sekme_uf:
    st.subheader("Giriş Parametrelerinin Tanım Alanları ve Anlık Üyelik Dereceleri (μ)")
    
    mu_d_k = fuzz.interp_membership(motor.evren_doluluk, motor.doluluk['Kritik_Dusuk'].mf, val_doluluk)
    mu_d_n = fuzz.interp_membership(motor.evren_doluluk, motor.doluluk['Nominal'].mf, val_doluluk)
    mu_d_t = fuzz.interp_membership(motor.evren_doluluk, motor.doluluk['Tasman_Sınırı'].mf, val_doluluk)

    mu_deb_k = fuzz.interp_membership(motor.evren_debi, motor.debi['Kurak'].mf, val_debi)
    mu_deb_n = fuzz.interp_membership(motor.evren_debi, motor.debi['Normal'].mf, val_debi)
    mu_deb_t = fuzz.interp_membership(motor.evren_debi, motor.debi['Taskın'].mf, val_debi)

    g1, g2, g3 = st.columns(3)
    
    with g1:
        fig1, ax1 = plt.subplots(figsize=(5, 3.2))
        ax1.plot(motor.evren_doluluk, motor.doluluk['Kritik_Dusuk'].mf, 'b', label='Kritik Düşük')
        ax1.plot(motor.evren_doluluk, motor.doluluk['Nominal'].mf, 'g', label='Nominal')
        ax1.plot(motor.evren_doluluk, motor.doluluk['Tasman_Sınırı'].mf, 'r', label='Taşma Sınırı')
        ax1.axvline(x=val_doluluk, color='purple', linestyle='--')
        ax1.set_title(f"Giriş 1: Baraj Doluluğu\nμKD: {mu_d_k:.2f} | μN: {mu_d_n:.2f} | μTS: {mu_d_t:.2f}", fontsize=8)
        ax1.grid(True, alpha=0.3); ax1.legend(fontsize=6)
        st.pyplot(fig1); plt.close(fig1)

    with g2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.plot(motor.evren_debi, motor.debi['Kurak'].mf, 'b', label='Kurak')
        ax2.plot(motor.evren_debi, motor.debi['Normal'].mf, 'g', label='Normal')
        ax2.plot(motor.evren_debi, motor.debi['Taskın'].mf, 'r', label='Taşkın')
        ax2.axvline(x=val_debi, color='purple', linestyle='--')
        ax2.set_title(f"Giriş 2: Nehir Akış Debisi\nμK: {mu_deb_k:.2f} | μN: {mu_deb_n:.2f} | μT: {mu_deb_t:.2f}", fontsize=8)
        ax2.grid(True, alpha=0.3); ax2.legend(fontsize=6)
        st.pyplot(fig2); plt.close(fig2)

    with g3:
        fig3, ax3 = plt.subplots(figsize=(5, 3.2))
        ax3.plot(motor.evren_fiyat, motor.fiyat['Ucuz'].mf, 'b', label='Ucuz')
        ax3.plot(motor.evren_fiyat, motor.fiyat['Standart'].mf, 'g', label='Standart')
        ax3.plot(motor.evren_fiyat, motor.fiyat['Pahalı'].mf, 'r', label='Pahalı')
        ax3.axvline(x=val_fiyat, color='purple', linestyle='--')
        ax3.set_title("Giriş 3: Elektrik Spot Fiyatı", fontsize=8)
        ax3.grid(True, alpha=0.3); ax3.legend(fontsize=6)
        st.pyplot(fig3); plt.close(fig3)

with sekme_inference:
    st.subheader("Mühendislik Kurallarının Aktivasyonu ve Alan Entegrasyonu")
    col_inf1, col_inf2 = st.columns(2)
    
    kural_gcleri = []
    aktif_kurallar_listesi = []
    TUM_KURAL_METNLERI = []

    # --- SÜPER GARANTİLİ STATİK VE DİNAMİK ALFA HESAPLAMA MATRİSİ ---
    # Kuralları bağımsız matematiksel diziler olarak el ile hesaplayıp simülasyon hatalarını bypass ediyoruz.
    sozluk_doluluk = {
        'Kritik_Dusuk': mu_d_k,
        'Nominal': mu_d_n,
        'Tasman_Sınırı': mu_d_t
    }
    sozluk_debi = {
        'Kurak': mu_deb_k,
        'Normal': mu_deb_n,
        'Taskın': mu_deb_t
    }
    
    mu_f_u = fuzz.interp_membership(motor.evren_fiyat, motor.fiyat['Ucuz'].mf, val_fiyat)
    mu_f_s = fuzz.interp_membership(motor.evren_fiyat, motor.fiyat['Standart'].mf, val_fiyat)
    mu_f_p = fuzz.interp_membership(motor.evren_fiyat, motor.fiyat['Pahalı'].mf, val_fiyat)
    
    sozluk_fiyat = {
        'Ucuz': mu_f_u,
        'Standart': mu_f_s,
        'Pahalı': mu_f_p
    }

    # 15 kuralın mantıksal şablon eşleşmesi
    sablon_kurallar = [
        ('Tasman_Sınırı', 'Taskın', None),
        ('Tasman_Sınırı', 'Normal', None),
        ('Nominal', 'Taskın', None),
        ('Kritik_Dusuk', 'Taskın', None),
        
        ('Tasman_Sınırı', None, 'Pahalı'),
        ('Nominal', None, 'Pahalı'),
        ('Nominal', None, 'Standart'),
        ('Kritik_Dusuk', None, 'Pahalı'),
        
        ('Kritik_Dusuk', None, 'Ucuz'),
        ('Kritik_Dusuk', None, 'Standart'),
        ('Nominal', None, 'Ucuz'),
        (None, 'Kurak', 'Ucuz'),
        
        ('Nominal', 'Normal', None),
        (None, 'Normal', 'Standart'),
        ('Tasman_Sınırı', None, 'Ucuz')
    ]

    for sira, sablon in enumerate(sablon_kurallar):
        degerler = []
        if sablon[0] is not None:
            degerler.append(sozluk_doluluk[sablon[0]])
        if sablon[1] is not None:
            degerler.append(sozluk_debi[sablon[1]])
        if sablon[2] is not None:
            degerler.append(sozluk_fiyat[sablon[2]])
            
        # Kesişim (VE) işlemi matematiksel minimumdur
        aktivasyon = min(degerler)
        kural_gcleri.append(aktivasyon)

        # Ham metin üretimi
        kural = motor.kurallar[sira]
        kural_metni = str(kural).replace("AND", "∧ (VE)").replace("THEN", "→ (İSE)").replace("IF", "EĞER")
        kural_metni = kural_metni.replace("Baraj_Doluluk", "Baraj_Doluluğu").replace("Gelen_Debi", "Nehir_Debisi").replace("Elektrik_Fiyati", "Elektrik_Fiyatı").replace("Sistem_Aktivasyon", "Sistem_Aktivasyonu")
        kural_metni = kural_metni.replace("Kritik_Dusuk", "Kritik_Düşük").replace("Tasman_Sınırı", "Taşma_Sınırı").replace("Taskın", "Taşkın")
        
        tam_metin = f"Kural {sira+1:02d}: {kural_metni}"
        TUM_KURAL_METNLERI.append(tam_metin)
        
        if aktivasyon > 0:
            aktif_kurallar_listesi.append((sira+1, tam_metin, aktivasyon))
            
    with col_inf1:
        st.write("**Grafik 4: Sektörel 15 Kuralın Alfa Kesim (alpha-cut) Dağılımı**")
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        x_etiketler = [f"K{i+1}" for i in range(15)]
        
        # Grafik çizimini doğrudan eldeki saf python listesinden (kural_gcleri) yapıyoruz
        ax4.bar(x_etiketler, kural_gcleri, color='teal', edgecolor='darkslategray', alpha=0.8)
        ax4.set_ylabel("Aktivasyon Derecesi (alpha)")
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, axis='y', linestyle=':', alpha=0.6)
        st.pyplot(fig4)
        plt.close(fig4)

    with col_inf2:
        st.write("**Grafik 5: Sentezlenmiş Karar Alanı ve İzdüşüm Noktası (Y*)**")
        fig5, ax5 = plt.subplots(figsize=(6, 3.5))
        ax5.plot(motor.evren_kontrol, motor.kontrol['Ekonomik_Koruma'].mf, 'b:', alpha=0.4, label='Eko Koruma')
        ax5.plot(motor.evren_kontrol, motor.kontrol['Dengeli_Uretim'].mf, 'g:', alpha=0.4, label='Dengeli Ür.')
        ax5.plot(motor.evren_kontrol, motor.kontrol['Maksimum_Tahliye'].mf, 'r:', alpha=0.4, label='Mak. Tahliye')
        
        if sim_aktuel:
            try:
                gercek_alan = motor.kontrol.testing_membership[sim_aktuel]
                ax5.fill_between(motor.evren_kontrol, 0, gercek_alan, facecolor='darkorange', alpha=0.4, label='Mamdani Çıktı Alanı')
                ax5.plot(motor.evren_kontrol, gercek_alan, 'darkorange', linewidth=2)
            except:
                pass
        
        ax5.axvline(x=cikis_v, color='black', linestyle='-', linewidth=2.5, label=f'Karar (%{cikis_v:.2f})')
        ax5.set_title(f"Kompozisyon Sentezi Modeli ({metot.upper()})", fontsize=9, fontweight='bold')
        ax5.legend(fontsize=7, loc='upper left')
        ax5.grid(True, alpha=0.2)
        st.pyplot(fig5)
        plt.close(fig5)

    # --- CANLI KURAL TAKİP MASASI ---
    st.markdown("---")
    st.subheader("Canlı Kural Takip Masası (SCADA Real-Time Inference)")
    st.markdown("Şu anki sensör değerleri ile tetiklenen **aktif kuralların listesi** ve etki güçleri aşağıda gösterilmektedir:")
    
    for no, metin, agirlik in aktif_kurallar_listesi:
        st.info(f" **[AKTİF - Etki Gücü: {agirlik:.2f}]** {metin}")

    st.markdown("---")
    st.write("**Grafik 6: Sistemin 3D Karar Yüzey Anatomisi**")
    
    x_mesh, y_mesh, z_mesh = hesapla_3d_baraj_yuzey(val_fiyat)
    fig6 = plt.figure(figsize=(10, 4.5))
    ax6 = fig6.add_subplot(111, projection='3d')
    surf = ax6.plot_surface(x_mesh, y_mesh, z_mesh, rstride=1, cstride=1, cmap='winter', edgecolor='none', alpha=0.85)
    ax6.set_xlabel('Baraj Doluluğu (%)', fontsize=8)
    ax6.set_ylabel('Gelen Debi (m³/s)', fontsize=8)
    ax6.set_zlabel('Sistem Aktivasyonu (%)', fontsize=8)
    fig6.colorbar(surf, shrink=0.5, aspect=8)
    st.pyplot(fig6)
    plt.close(fig6)

# --- TÜM LİSTE ARŞİVİ ---
st.markdown("---")
st.markdown("###  SCADA Otomasyon Kural Kataloğu (Tüm Liste)")
with st.expander("Sistemde Tanımlı Olan 15 Ana Kuralın Tamamını İnceleyin"):
    for m in TUM_KURAL_METNLERI:
        st.code(m, language="text")