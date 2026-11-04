import customtkinter as ctk
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

# Memastikan matplotlib menggunakan backend TkAgg agar kompatibel dengan Tkinter
matplotlib.use('TkAgg')

# ==========================================
# BAGIAN 1: MODEL (Logika Bisnis & Machine Learning)
# ==========================================
class ModelPrediksi:
    def __init__(self):
        # DATA DUMMY: Ganti dengan data riil dari BPS atau Dinas Pariwisata Pematangsiantar
        # X = Tahun, y = Jumlah Wisatawan (dalam ribuan) atau Okupansi Hotel (%)
        self.X = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]).reshape(-1, 1)
        self.y = np.array([120, 135, 150, 180, 210, 85, 110, 175, 240, 270, 300]) 
        
        # Model Regresi Polinomial Derajat 2 (Quadratic)
        # Derajat 2 cocok untuk pola pertumbuhan yang memiliki akselerasi atau perlambatan
        self.model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        self.model.fit(self.X, self.y)

    def prediksi(self, tahun):
        """Mengembalikan nilai prediksi untuk tahun tertentu"""
        return self.model.predict([[tahun]])[0]

    def get_data(self):
        """Mengembalikan data historis untuk keperluan plotting grafik"""
        return self.X, self.y

    def get_trend_line(self):
        """Menghasilkan data garis tren untuk rentang tahun 2015-2030"""
        X_range = np.linspace(2015, 2030, 100).reshape(-1, 1)
        y_range = self.model.predict(X_range)
        return X_range, y_range


# ==========================================
# BAGIAN 2: VIEW & CONTROLLER (Antarmuka & Event Handling)
# ==========================================
class AplikasiSiantar(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Konfigurasi Jendela Utama
        self.title("Sistem Prediksi Pariwisata & Perhotelan - Kota Pematangsiantar")
        self.geometry("850x650")
        self.minsize(800, 600)
        
        # Pengaturan Tema CustomTkinter
        ctk.set_appearance_mode("System")  # Mengikuti mode sistem (Light/Dark)
        ctk.set_default_color_theme("blue") # Tema warna biru profesional

        # Inisialisasi Model
        self.model = ModelPrediksi()
        
        # Bangun Antarmuka
        self.setup_ui()

    def setup_ui(self):
        # 1. Header
        self.header = ctk.CTkLabel(self, text="Sistem Prediksi Laju Pertumbuhan\nPariwisata & Perhotelan Pematangsiantar", 
                                   font=("Segoe UI", 22, "bold"))
        self.header.pack(pady=15)

        # 2. Frame Input (Menggunakan Layout Grid untuk kerapian)
        self.input_frame = ctk.CTkFrame(self, corner_radius=10)
        self.input_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(self.input_frame, text="Masukkan Tahun Prediksi:", font=("Segoe UI", 14)).grid(row=0, column=0, padx=15, pady=15, sticky="w")
        
        self.entry_tahun = ctk.CTkEntry(self.input_frame, placeholder_text="Contoh: 2027", width=150, font=("Segoe UI", 14))
        self.entry_tahun.grid(row=0, column=1, padx=10, pady=15)

        self.btn_prediksi = ctk.CTkButton(self.input_frame, text="🔮 Hitung Prediksi", command=self.proses_prediksi, 
                                          font=("Segoe UI", 14, "bold"), fg_color="#2b7bb9", hover_color="#1e5a8a")
        self.btn_prediksi.grid(row=0, column=2, padx=15, pady=15)

        # 3. Frame Hasil & Visualisasi
        self.hasil_frame = ctk.CTkFrame(self, corner_radius=10)
        self.hasil_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Label Hasil Teks
        self.label_hasil = ctk.CTkLabel(self.hasil_frame, text="Silakan masukkan tahun dan klik 'Hitung Prediksi'", 
                                        font=("Segoe UI", 16, "bold"), text_color="gray")
        self.label_hasil.pack(pady=10)

        # Frame Khusus untuk Grafik Matplotlib
        self.grafik_frame = ctk.CTkFrame(self.hasil_frame)
        self.grafik_frame.pack(pady=5, padx=15, fill="both", expand=True)

        # 4. Footer HAKI (Penting untuk Tinjauan Komersial)
        self.footer = ctk.CTkLabel(self, text="© 2026 Hak Cipta Dilindungi Undang-Undang (UU No. 28 Tahun 2014). Dilarang memperbanyak atau mendistribusikan tanpa izin tertulis.", 
                                   text_color="gray", font=("Segoe UI", 10))
        self.footer.pack(pady=5)

        # Tampilkan grafik awal (tren historis)
        self.update_grafik()

    # --- CONTROLLER: Mekanisme Event-Driven & Callback ---
    def proses_prediksi(self):
        """Callback function yang dipicu saat tombol diklik"""
        try:
            # Validasi Input
            tahun_input = self.entry_tahun.get()
            if not tahun_input.isdigit():
                raise ValueError("Input harus berupa angka tahun (contoh: 2027)")
            
            tahun = int(tahun_input)
            if tahun < 2015 or tahun > 2035:
                raise ValueError("Tahun di luar rentang prediksi yang valid (2015 - 2035)")
            
            # UX: Tampilkan status memproses
            self.label_hasil.configure(text="⏳ Memproses model regresi...", text_color="orange")
            self.update() # Paksa refresh UI agar teks terlihat sebelum proses berat (jika ada)

            # Eksekusi Model
            hasil_prediksi = self.model.prediksi(tahun)
            
            # Update UI dengan hasil
            self.label_hasil.configure(
                text=f"✅ Prediksi Tahun {tahun}: {hasil_prediksi:.2f} Ribu Wisatawan", 
                text_color="#2ecc71" # Warna hijau sukses
            )
            
            # Update grafik dengan titik prediksi baru
            self.update_grafik(tahun_prediksi=tahun, nilai_prediksi=hasil_prediksi)

        except ValueError as e:
            # Penanganan Error yang User-Friendly
            self.label_hasil.configure(text=f"⚠️ Error: {str(e)}", text_color="#e74c3c")

    def update_grafik(self, tahun_prediksi=None, nilai_prediksi=None):
        """Menggambar ulang grafik di dalam frame Tkinter"""
        # Bersihkan grafik lama jika ada
        for widget in self.grafik_frame.winfo_children():
            widget.destroy()

        # Buat figure matplotlib
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
        
        # ✅ PERBAIKAN: Gunakan tuple RGBA (0,0,0,0) untuk transparansi
        # Format: (Red, Green, Blue, Alpha) - Alpha=0 berarti transparan 100%
        fig.patch.set_facecolor((0, 0, 0, 0))
        ax.set_facecolor((0, 0, 0, 0))

        X_hist, y_hist = self.model.get_data()
        X_trend, y_trend = self.model.get_trend_line()
        
        # Plot 1: Data Historis (Scatter)
        ax.scatter(X_hist, y_hist, color='#3498db', s=60, label='Data Historis (BPS/Dinas)', zorder=3)
        
        # Plot 2: Garis Tren Regresi Polinomial
        ax.plot(X_trend, y_trend, color='#e74c3c', linestyle='--', linewidth=2, label='Tren Regresi Polinomial (Derajat 2)', zorder=2)

        # Plot 3: Titik Prediksi (Jika user sudah melakukan prediksi)
        if tahun_prediksi and nilai_prediksi:
            ax.scatter([tahun_prediksi], [nilai_prediksi], color='#2ecc71', s=120, zorder=4, 
                       edgecolor='white', label=f'Prediksi {tahun_prediksi}')

        # Konfigurasi Estetika Grafik
        ax.set_title("Tren Pertumbuhan Wisatawan Pematangsiantar", fontweight='bold', pad=10)
        ax.set_xlabel("Tahun", fontweight='bold')
        ax.set_ylabel("Jumlah (Ribu Orang)", fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Sesuaikan warna teks grafik agar terlihat di Dark Mode
        text_color = 'white' if ctk.get_appearance_mode() == "Dark" else 'black'
        ax.tick_params(colors=text_color)
        ax.title.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)

        # Embed Matplotlib ke dalam CustomTkinter Frame
        canvas = FigureCanvasTkAgg(fig, master=self.grafik_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ==========================================
# BAGIAN 3: EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    # Jalankan aplikasi
    app = AplikasiSiantar()
    app.mainloop()