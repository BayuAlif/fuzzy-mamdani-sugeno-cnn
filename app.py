import streamlit as st
import os
from PIL import Image

#import fungsi backend dari file fuzzy_logic.py
from fuzzy_logic import (
    dapatkan_skor_penyakit_cnn,
    fuzzifikasi_penyakit,
    fuzzifikasi_suhu,
    fuzzifikasi_kelembapan,
    fuzzifikasi_hujan,
    fuzzifikasi_umur,
    inferensi,
    agregasi_aturan,
    defuzzifikasi_sugeno,
    defuzzifikasi_mamdani
)

#Konfigurasi streamlit
st.set_page_config(page_title="Sistem Deteksi Gagal Panen", page_icon="🌱", layout="centered")

st.title("🌱 Sistem Penentuan Risiko Gagal Panen")
st.markdown("""
Aplikasi ini mengintegrasikan **CNN (Deep Learning)** untuk deteksi penyakit tanaman 
dan **Fuzzy Logic (Mamdani & Sugeno)** untuk memprediksi risiko gagal panen berdasarkan faktor lingkungan.
""")

st.divider()

#1. INPUT GAMBAR (CNN)
st.header("1. Upload Gambar Daun")
uploaded_file = st.file_uploader("Pilih gambar daun (format .jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

skor_cnn = 0.0
kelas_terdeteksi = "Belum ada gambar"

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Daun yang Diupload", use_container_width=True)
    
    # Simpan sementara untuk dibaca oleh PyTorch
    temp_path = os.path.join("temp_image.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner('Model CNN sedang menganalisis penyakit...'):
        # Jalankan prediksi CNN
        skor_cnn, kelas_terdeteksi, akurasi = dapatkan_skor_penyakit_cnn(temp_path)
    
    st.success("Deteksi Selesai!")
    
    nama_kelas_bersih = kelas_terdeteksi.replace('_', ' ').replace('  ', ' ').strip()
    st.info(f"🧬 **Prediksi CNN:** {nama_kelas_bersih}")
    
    col1, col2 = st.columns(2)
    col1.metric("Confidence Model", f"{akurasi:.2f}%")
    col2.metric("Skor Input Fuzzy", f"{skor_cnn:.2f}%")

st.divider()

#2. INPUT LINGKUNGAN (SLIDER)
st.header("2. Input Parameter Lingkungan")
st.write("Geser slider di bawah ini sesuai dengan data kondisi lingkungan nyata.")

col1, col2 = st.columns(2)
with col1:
    suhu_input = st.slider("🌡️ Suhu Udara (°C)", min_value=10.0, max_value=40.0, value=25.0, step=0.5)
    kelembapan_input = st.slider("💧 Kelembapan Tanah (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
with col2:
    hujan_input = st.slider("🌧️ Curah Hujan (mm/hari)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
    umur_input = st.slider("⏳ Umur Tanaman (Hari)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)

st.divider()

#3. EKSEKUSI FUZZY LOGIC
st.header("3. Hasil Prediksi Risiko")

# Tombol untuk mengeksekusi sistem
if st.button("🚀 Analisis Risiko Gagal Panen", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ Mohon upload gambar daun terlebih dahulu!")
    else:
        with st.spinner('Mengeksekusi Rule Base Fuzzy...'):
            # 1. Fuzzifikasi
            f_penyakit = fuzzifikasi_penyakit(skor_cnn)
            f_suhu = fuzzifikasi_suhu(suhu_input)
            f_kelembapan = fuzzifikasi_kelembapan(kelembapan_input)
            f_hujan = fuzzifikasi_hujan(hujan_input)
            f_umur = fuzzifikasi_umur(umur_input)
            
            # 2. Inferensi
            evaluasi_rules = inferensi(f_penyakit, f_suhu, f_kelembapan, f_hujan, f_umur)
            hasil_akhir_inferensi = agregasi_aturan(evaluasi_rules)
            
            # 3. Defuzzifikasi
            skor_sugeno, label_sugeno = defuzzifikasi_sugeno(hasil_akhir_inferensi)
            skor_mamdani, label_mamdani = defuzzifikasi_mamdani(hasil_akhir_inferensi)
            
            if skor_sugeno == 0.0 and skor_mamdani == 0.0:
                st.warning("⚠️ Kondisi lingkungan yang diinputkan tidak terdefinisi dalam 18 Rule Base saat ini (Blind Spot). Silakan sesuaikan slider atau tambahkan rule baru di backend.")
            else:
                # Tampilkan Hasil Perbandingan jika ada nilainya
                st.subheader("Perbandingan Output Defuzzifikasi")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.info("### Metode Sugeno")
                    st.metric("Skor Risiko Akhir", f"{skor_sugeno} / 100")
                    st.metric("Status", label_sugeno)
                    
                with res_col2:
                    st.success("### Metode Mamdani")
                    st.metric("Skor Risiko Akhir", f"{skor_mamdani} / 100")
                    st.metric("Status", label_mamdani)
        
            # Bersihkan file gambar sementara
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")