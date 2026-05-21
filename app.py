import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skfuzzy as fuzz
import random

class EndustriyelBarajKontrolMotoru:
    def __init__(self):
        self.evren_doluluk = np.arange(0, 101, 1)
        self.evren_debi    = np.arange(0, 1001, 1)
        self.evren_fiyat   = np.arange(0, 5001, 1)
        self.evren_kontrol = np.arange(0, 101, 1)
        self.bulanik_mimarini_kur()

    def bulanik_mimarini_kur(self):
        self.mf_doluluk = [fuzz.trapmf(self.evren_doluluk, [0,0,35,55]), fuzz.trimf(self.evren_doluluk, [35,55,75]), fuzz.trapmf(self.evren_doluluk, [55,75,100,100])]
        self.mf_debi    = [fuzz.trapmf(self.evren_debi, [0,0,250,450]), fuzz.trimf(self.evren_debi, [250,500,750]), fuzz.trapmf(self.evren_debi, [550,750,1000,1000])]
        self.mf_fiyat   = [fuzz.trapmf(self.evren_fiyat, [0,0,1500,2500]), fuzz.trimf(self.evren_fiyat, [1500,2700,3800]), fuzz.trapmf(self.evren_fiyat, [2800,4000,5000,5000])]
        self.mf_kontrol = [fuzz.trimf(self.evren_kontrol, [0,0,45]), fuzz.trimf(self.evren_kontrol, [30,50,70]), fuzz.trapmf(self.evren_kontrol, [55,75,100,100])]

    def kurallari_degerlendir(self, v1, v2, v3):
        mu_d = [fuzz.interp_membership(self.evren_doluluk, m, v1) for m in self.mf_doluluk]
        mu_deb = [fuzz.interp_membership(self.evren_debi, m, v2) for m in self.mf_debi]
        mu_f = [fuzz.interp_membership(self.evren_fiyat, m, v3) for m in self.mf_fiyat]
        
        guc = [
            min(mu_d[2], mu_deb[2]), min(mu_d[2], mu_deb[1]), min(mu_d[1], mu_deb[2]),
            min(mu_d[0], mu_deb[2]), min(mu_d[2], mu_f[2]),   min(mu_d[1], mu_f[2]),
            min(mu_d[1], mu_f[1]),   min(mu_d[0], mu_f[2]),   min(mu_d[0], mu_f[0]),
            min(mu_d[0], mu_f[1]),   min(mu_d[1], mu_f[0]),   min(mu_deb[0], mu_f[0]),
            min(mu_d[1], mu_deb[1]), min(mu_deb[1], mu_f[1]), min(mu_d[2], mu_f[0])
        ]
        
        birlesik = np.zeros_like(self.evren_kontrol)
        for i in range(15):
            kural_sonucu = np.fmin(guc[i], self.mf_kontrol[i % 3])
            birlesik = np.fmax(birlesik, kural_sonucu)
        
        if np.sum(birlesik) == 0:
            cikis_v = 0
        else:
            cikis_v = fuzz.defuzz(self.evren_kontrol, birlesik, 'centroid')
        return cikis_v, guc, birlesik

class HESProDashboard:
    def __init__(self, root):
        self.motor = EndustriyelBarajKontrolMotoru()
        self.root = root
        self.root.title("HES Akıllı Otomasyon | SCADA Karar Destek Sistemi PRO")
        self.pencereyi_ortala(1250, 750)
        self.root.configure(bg="#1e1e1e")
        self.simulasyon_aktif = False
        self.toplam_kazanc = 0.0
        self.aciklamalar = [
            "Yüksek Doluluk + Yüksek Debi -> Maksimum Üretim", "Yüksek Doluluk + Orta Debi -> Yüksek Üretim",
            "Orta Doluluk + Yüksek Debi -> Yüksek Üretim", "Düşük Doluluk + Yüksek Debi -> Orta Üretim",
            "Yüksek Doluluk + Yüksek Fiyat -> Maksimum Kâr Modu", "Orta Doluluk + Yüksek Fiyat -> Yüksek Kâr Modu",
            "Orta Doluluk + Orta Fiyat -> Nominal Üretim", "Düşük Doluluk + Yüksek Fiyat -> Orta Üretim",
            "Düşük Doluluk + Düşük Fiyat -> Minumum Üretim", "Düşük Doluluk + Orta Fiyat -> Düşük Üretim",
            "Orta Doluluk + Düşük Fiyat -> Düşük Üretim", "Düşük Debi + Düşük Fiyat -> Bekleme Modu",
            "Orta Doluluk + Orta Debi -> Nominal Üretim", "Orta Debi + Orta Fiyat -> Nominal Üretim",
            "Yüksek Doluluk + Düşük Fiyat -> Düşük Üretim"
        ]

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d2d2d", foreground="#aaaaaa", borderwidth=1, padding=[12, 4], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#007acc")], foreground=[("selected", "white")])
        style.configure("TFrame", background="#1e1e1e")

        self.control_frame = tk.Frame(root, bg="#2d2d2d", bd=1, relief=tk.SOLID)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(self.control_frame, text="SİSTEM GİRİŞLERİ", bg="#2d2d2d", fg="#007acc", font=("Segoe UI", 12, "bold")).pack(pady=(10,15))

        self.s1 = self.create_custom_scale("Doluluk Oranı (%)", 0, 100)
        self.s2 = self.create_custom_scale("Su Debisi (m³/s)", 0, 1000)
        self.s3 = self.create_custom_scale("Elektrik Fiyatı (TL/MWh)", 0, 5000)
        
        self.btn_analiz = tk.Button(self.control_frame, text="MANUEL ANALİZ ET", command=lambda: self.hesapla(otomatik=False), bg="#4e4e4e", fg="white", activebackground="#3a3a3a", activeforeground="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", pady=6, padx=10)
        self.btn_analiz.pack(fill=tk.X, padx=15, pady=(15,5))

        self.btn_sim = tk.Button(self.control_frame, text="CANLI SİMÜLASYONU BAŞLAT", command=self.toggle_simulasyon, bg="#007acc", fg="white", activebackground="#005999", activeforeground="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", pady=8, padx=10)
        self.btn_sim.pack(fill=tk.X, padx=15, pady=5)
        
        self.alarm_frame = tk.Frame(self.control_frame, bg="#333333", height=40, bd=1, relief=tk.SUNKEN)
        self.alarm_frame.pack(fill=tk.X, padx=15, pady=10)
        self.alarm_frame.pack_propagate(False)
        self.lbl_alarm = tk.Label(self.alarm_frame, text="SİSTEM DURUMU: NORMAL", bg="#333333", fg="#00ff66", font=("Segoe UI", 9, "bold"))
        self.lbl_alarm.pack(fill=tk.BOTH, expand=True)

        self.create_output_cards()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_dashboard = ttk.Frame(self.notebook)
        self.tab_mf_graphs = ttk.Frame(self.notebook)
        self.tab_rules = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_dashboard, text=" Canlı SCADA Ekranı ")
        self.notebook.add(self.tab_mf_graphs, text=" Üyelik Fonksiyonları (MF) ")
        self.notebook.add(self.tab_rules, text=" Kural Matrisi / Veri Tabanı ")

        self.top_graph_frame = tk.Frame(self.tab_dashboard, bg="#1e1e1e")
        self.top_graph_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 3.5), facecolor="#1e1e1e")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top_graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_frame = tk.LabelFrame(self.tab_dashboard, text=" Aktif SCADA Karar Raporu ve Sistem Günlüğü ", bg="#1e1e1e", fg="#aaaaaa", font=("Segoe UI", 9))
        self.log_frame.pack(fill=tk.X, padx=5, pady=5)
        self.log = scrolledtext.ScrolledText(self.log_frame, height=7, bg="#252526", fg="#00FF00", insertbackground="white", font=("Consolas", 10), bd=0)
        self.log.pack(fill=tk.X, padx=5, pady=5)
        
        self.setup_mf_graphs_tab()
        self.setup_rules_tab()
        self.hesapla(otomatik=False)

    def create_output_cards(self):
        c1 = tk.Frame(self.control_frame, bg="#252526", bd=1, relief=tk.RAISED)
        c1.pack(fill=tk.X, padx=15, pady=4)
        tk.Label(c1, text="HES VALF ÇIKTI DEĞERİ", bg="#252526", fg="#888888", font=("Segoe UI", 8, "bold")).pack(pady=(3,0))
        self.lbl_result = tk.Label(c1, text="--", bg="#252526", fg="#00ff66", font=("Segoe UI", 16, "bold"))
        self.lbl_result.pack(pady=(0,3))

        c2 = tk.Frame(self.control_frame, bg="#252526", bd=1, relief=tk.RAISED)
        c2.pack(fill=tk.X, padx=15, pady=4)
        tk.Label(c2, text="ANLIK ÜRETİM GÜCÜ", bg="#252526", fg="#888888", font=("Segoe UI", 8, "bold")).pack(pady=(3,0))
        self.lbl_power = tk.Label(c2, text="0.00 MW", bg="#252526", fg="#00bcff", font=("Segoe UI", 16, "bold"))
        self.lbl_power.pack(pady=(0,3))

        c3 = tk.Frame(self.control_frame, bg="#252526", bd=1, relief=tk.RAISED)
        c3.pack(fill=tk.X, padx=15, pady=4)
        tk.Label(c3, text="TOPLAM FİNANSAL CİRO", bg="#252526", fg="#888888", font=("Segoe UI", 8, "bold")).pack(pady=(3,0))
        self.lbl_finance = tk.Label(c3, text="0.00 TL", bg="#252526", fg="#ffcc00", font=("Segoe UI", 16, "bold"))
        self.lbl_finance.pack(pady=(0,3))

    def setup_mf_graphs_tab(self):
        self.fig_mf, self.axs_mf = plt.subplots(3, 1, figsize=(6, 6), facecolor="#1e1e1e")
        names = ["Baraj Doluluk Kümesi", "Su Debisi Kümesi", "Elektrik Fiyat Kümesi"]
        universes = [self.motor.evren_doluluk, self.motor.evren_debi, self.motor.evren_fiyat]
        mfs = [self.motor.mf_doluluk, self.motor.mf_debi, self.motor.mf_fiyat]
        colors = [['#ff5555', '#55ff55', '#5555ff'], ['#ffaa00', '#00ffcc', '#cc00ff'], ['#ffff55', '#ff55aa', '#00aaff']]
        labels = ["Düşük", "Orta", "Yüksek"]

        for idx, ax in enumerate(self.axs_mf):
            ax.set_facecolor("#252526")
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, color="#3a3a3a", alpha=0.4)
            ax.set_title(names[idx], color="white", fontsize=9, fontweight='bold')
            for i in range(3):
                ax.plot(universes[idx], mfs[idx][i], color=colors[idx][i], linewidth=2, label=labels[i])
            ax.legend(facecolor='#2d2d2d', edgecolor='none', labelcolor='white', fontsize=7, loc='upper right')
        
        self.fig_mf.tight_layout()
        canvas_mf = FigureCanvasTkAgg(self.fig_mf, master=self.tab_mf_graphs)
        canvas_mf.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def toggle_simulasyon(self):
        self.simulasyon_aktif = not self.simulasyon_aktif
        if self.simulasyon_aktif:
            self.btn_sim.config(text="SİMÜLASYONU DURDUR", bg="#ff3366")
            self.btn_analiz.config(state=tk.DISABLED)
            self.animasyon_dongusu()
        else:
            self.btn_sim.config(text="CANLI SİMÜLASYONU BAŞLAT", bg="#007acc")
            self.btn_analiz.config(state=tk.NORMAL)

    def animasyon_dongusu(self):
        if not self.simulasyon_aktif: return
        self.s1.set(max(0, min(100, self.s1.get() + random.choice([-1, 0, 1]))))
        self.s2.set(max(0, min(1000, self.s2.get() + random.randint(-40, 40))))
        self.s3.set(max(0, min(5000, self.s3.get() + random.randint(-200, 200))))
        self.hesapla(otomatik=True)
        self.root.after(1000, self.animasyon_dongusu)

    def alarm_kontrol(self, doluluk, debi, valf):
        if doluluk > 85 and debi > 800:
            self.alarm_frame.config(bg="#ff3333")
            self.lbl_alarm.config(text="KRİTİK ALARM: TAŞKIN TEHLİKESİ!", bg="#ff3333", fg="white")
        elif doluluk < 15 and valf > 40:
            self.alarm_frame.config(bg="#ffcc00")
            self.lbl_alarm.config(text="UYARI: DÜŞÜK SEVİYE / KAVİTASYON RİSKİ", bg="#ffcc00", fg="black")
        else:
            self.alarm_frame.config(bg="#333333")
            self.lbl_alarm.config(text="SİSTEM DURUMU: NOMİNAL OPERASYON", bg="#333333", fg="#00ff66")

    def hesapla(self, otomatik=False):
        if not otomatik and self.simulasyon_aktif: return
        d_val = self.s1.get()
        deb_val = self.s2.get()
        f_val = self.s3.get()
        cikis, guc, alan = self.motor.kurallari_degerlendir(d_val, deb_val, f_val)
        self.lbl_result.config(text=f"% {cikis:.2f}")
        
        anlik_guc = (deb_val / 1000.0) * (cikis / 100.0) * 120.0
        anlik_kazanc = (anlik_guc * f_val) / 3600.0
        
        if not otomatik:
            self.toplam_kazanc += anlik_kazanc
            
        self.lbl_power.config(text=f"{anlik_guc:.2f} MW")
        self.lbl_finance.config(text=f"{self.toplam_kazanc:,.2f} TL")
        self.alarm_kontrol(d_val, deb_val, cikis)
        self.ax.clear()
        self.ax.set_facecolor("#252526")
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.grid(True, color="#3a3a3a", linestyle="--", alpha=0.5)
        self.ax.plot(self.motor.evren_kontrol, alan, color='#00ff66', linewidth=2.5, label='Bulanık Karar Alanı')
        self.ax.fill_between(self.motor.evren_kontrol, alan, color='#00ff66', alpha=0.15)
        self.ax.axvline(x=cikis, color='#ff3366', linestyle='--', linewidth=2, label=f'Ağırlık Merkezi: {cikis:.1f}')
        self.ax.set_title("Bulanık Kontrol Çıktı Kümesi Görselleştirmesi", color="white", fontsize=11, fontweight='bold', pad=10)
        self.ax.set_xlabel("Valf Açıklık Oranı (%)", color="#aaaaaa", fontsize=9)
        self.ax.set_ylabel("Üyelik Derecesi (μ)", color="#aaaaaa", fontsize=9)
        self.ax.legend(facecolor='#2d2d2d', edgecolor='none', labelcolor='white', loc='upper left')
        self.fig.tight_layout()
        self.canvas.draw()
        self.log.delete(1.0, tk.END)
        self.log.insert(tk.END, f"[SCADA VERİSİ] Doluluk: %{d_val} | Debi: {deb_val} m³/s | Fiyat: {f_val} TL/MWh\n")
        self.log.insert(tk.END, f"[ÜRETİM]       Aktif Çıkış Gücü: {anlik_guc:.2f} MW | Saniyelik Kazanç Oranı: {anlik_kazanc:.4f} TL/sn\n")
        self.log.insert(tk.END, "---------------------------------------------------------------------------------\n")
        for i, g in enumerate(guc):
            if g > 0.05:
                self.log.insert(tk.END, f" ✔ [K{i+1:02d}] Aktivasyon Derecesi: {g:.2f} | {self.aciklamalar[i]}\n")

    def pencereyi_ortala(self, genislik, yukseklik):
        ekran_genislik = self.root.winfo_screenwidth()
        ekran_yukseklik = self.root.winfo_screenheight()
        x = (ekran_genislik // 2) - (genislik // 2)
        y = (ekran_yukseklik // 2) - (yukseklik // 2)
        self.root.geometry(f"{genislik}x{yukseklik}+{x}+{y}")

    def create_custom_scale(self, label, min_v, max_v):
        frame = tk.Frame(self.control_frame, bg="#2d2d2d")
        frame.pack(fill=tk.X, padx=15, pady=6)
        tk.Label(frame, text=label, bg="#2d2d2d", fg="white", font=("Segoe UI", 10)).pack(anchor="w")
        scale = tk.Scale(frame, from_=min_v, to=max_v, orient='horizontal', bg="#2d2d2d", fg="#aaaaaa", troughcolor="#1e1e1e", highlightbackground="#2d2d2d", activebackground="#007acc", bd=0, font=("Segoe UI", 9))
        scale.pack(fill=tk.X, pady=(2,0))
        scale.bind("<ButtonRelease-1>", lambda event: self.hesapla(otomatik=False))
        return scale

    def setup_rules_tab(self):
        self.tree = ttk.Treeview(self.tab_rules, columns=("kural_no", "kural_tanimi", "cikis_katman"), show="headings", height=16)
        self.tree.heading("kural_no", text="Kural ID"); self.tree.heading("kural_tanimi", text="Eğer Koşulu"); self.tree.heading("cikis_katman", text="O Halde")
        self.tree.column("kural_no", width=80, anchor="center"); self.tree.column("kural_tanimi", width=500, anchor="w"); self.tree.column("cikis_katman", width=180, anchor="center")
        for i, aciklama in enumerate(self.aciklamalar):
            parts = aciklama.split(" -> ")
            self.tree.insert("", tk.END, values=(f"Kural {i+1}", parts[0], parts[1] if len(parts)>1 else ""))
        self.tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = HESProDashboard(root)
    root.mainloop()