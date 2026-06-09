import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os



# 1. ARSITEKTUR CNN DARI MAIN_STRATIFIED
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def dapatkan_skor_penyakit_cnn(image_path, model_path="model_stra.pth"):
    class_names = [
        'Pepper_bell__Bacterial_spot', 'Pepper_bell__healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
        'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
        'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 
        'Tomato__Tomato_YellowLeaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
    ]
    kelas_sehat = ['Pepper_bell__healthy', 'Potato___healthy', 'Tomato_healthy']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantCNN(num_classes=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        nama_prediksi = class_names[pred.item()]
        persentase_yakin = confidence.item() * 100

        if nama_prediksi in kelas_sehat:
            return 0.0, nama_prediksi, persentase_yakin
        else:
            return persentase_yakin, nama_prediksi, persentase_yakin


# 2. FUZZIFIKASI (DERAJAT KEANGGOTAAN)
def fungsi_segitiga(x, a, b, c):
    if x == b:
        return 1.0
    elif x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0.0

def fuzzifikasi_penyakit(x):
    return {
        "Ringan": fungsi_segitiga(x, 0, 0, 50),
        "Sedang": fungsi_segitiga(x, 25, 50, 75),
        "Parah": fungsi_segitiga(x, 50, 100, 100)
    } 

def fuzzifikasi_suhu(x):
    return {
        "Dingin": fungsi_segitiga(x, 10, 10, 25),
        "Normal": fungsi_segitiga(x, 20, 25, 30),
        "Panas":  fungsi_segitiga(x, 25, 40, 40)
    }
    
def fuzzifikasi_kelembapan(x):
    return {
        "Kering": fungsi_segitiga(x, 0, 0, 40),
        "Ideal":  fungsi_segitiga(x, 30, 50, 70),
        "Basah":  fungsi_segitiga(x, 60, 100, 100)
    }
    
def fuzzifikasi_hujan(x):
    return {
        "Rendah": fungsi_segitiga(x, 0, 0, 20),
        "Sedang": fungsi_segitiga(x, 10, 25, 40),
        "Tinggi": fungsi_segitiga(x, 30, 50, 50)
    }

def fuzzifikasi_umur(x):
    return {
        "Awal":   fungsi_segitiga(x, 0, 0, 45),
        "Tengah": fungsi_segitiga(x, 30, 60, 90),
        "Akhir":  fungsi_segitiga(x, 75, 120, 120)
    }


# 3. INFERENSI (RULE BASE)
def inferensi(f_penyakit, f_suhu, f_kelembapan, f_hujan, f_umur):
    rules_aktif = []

    alpha_1 = min(f_penyakit['Parah'], f_suhu['Panas'], f_kelembapan['Kering'], f_hujan['Rendah'], f_umur['Akhir'])
    rules_aktif.append({'output': 'Bahaya', 'alpha': alpha_1})

    alpha_2 = min(f_penyakit['Parah'], f_suhu['Normal'], f_kelembapan['Basah'], f_hujan['Tinggi'], f_umur['Tengah'])
    rules_aktif.append({'output': 'Bahaya', 'alpha': alpha_2})

    alpha_3 = min(f_penyakit['Ringan'], f_suhu['Normal'], f_kelembapan['Ideal'], f_hujan['Sedang'], f_umur['Awal'])
    rules_aktif.append({'output': 'Aman', 'alpha': alpha_3})

    alpha_4 = min(f_penyakit['Sedang'], f_suhu['Panas'], f_kelembapan['Kering'], f_hujan['Rendah'], f_umur['Tengah'])
    rules_aktif.append({'output': 'Waspada', 'alpha': alpha_4})

    alpha_5 = min(f_penyakit['Sedang'], f_suhu['Dingin'], f_kelembapan['Basah'], f_hujan['Tinggi'], f_umur['Akhir'])
    rules_aktif.append({'output': 'Waspada', 'alpha': alpha_5})

    alpha_6 = min(f_penyakit['Parah'], f_suhu['Dingin'], f_kelembapan['Basah'], f_hujan['Sedang'], f_umur['Awal'])
    rules_aktif.append({'output': 'Bahaya', 'alpha': alpha_6})

    alpha_7 = min(f_penyakit['Ringan'], f_suhu['Normal'], f_kelembapan['Kering'], f_hujan['Rendah'], f_umur['Tengah'])
    rules_aktif.append({'output': 'Waspada', 'alpha': alpha_7})

    alpha_8 = min(f_penyakit['Sedang'], f_suhu['Normal'], f_kelembapan['Ideal'], f_hujan['Sedang'], f_umur['Akhir'])
    rules_aktif.append({'output': 'Aman', 'alpha': alpha_8})

    alpha_9 = min(f_penyakit['Parah'], f_suhu['Normal'], f_kelembapan['Ideal'], f_hujan['Sedang'], f_umur['Tengah'])
    rules_aktif.append({'output': 'Bahaya', 'alpha': alpha_9})

    alpha_10 = min(f_penyakit['Ringan'], f_suhu['Panas'], f_kelembapan['Kering'], f_hujan['Rendah'], f_umur['Awal'])
    rules_aktif.append({'output': 'Waspada', 'alpha': alpha_10})

    alpha_11 = min(f_penyakit['Sedang'], f_suhu['Dingin'], f_kelembapan['Ideal'], f_hujan['Tinggi'], f_umur['Tengah'])
    rules_aktif.append({'output': 'Waspada', 'alpha': alpha_11})

    alpha_12 = min(f_penyakit['Ringan'], f_suhu['Dingin'], f_kelembapan['Basah'], f_hujan['Tinggi'], f_umur['Akhir'])
    rules_aktif.append({'output': 'Waspada', 'alpha': alpha_12})

    alpha_13 = min(f_penyakit['Parah'], f_suhu['Panas'], f_kelembapan['Basah'], f_hujan['Tinggi'], f_umur['Awal'])
    rules_aktif.append({'output': 'Bahaya', 'alpha': alpha_13})

    alpha_14 = min(f_penyakit['Sedang'], f_suhu['Panas'], f_kelembapan['Ideal'], f_hujan['Rendah'], f_umur['Akhir'])
    rules_aktif.append({'output': 'Waspada', 'alpha': alpha_14})

    alpha_15 = min(f_penyakit['Ringan'], f_suhu['Normal'], f_kelembapan['Ideal'], f_hujan['Rendah'], f_umur['Akhir'])
    rules_aktif.append({'output': 'Aman', 'alpha': alpha_15})

    return rules_aktif

def agregasi_aturan(rules_aktif):
    hasil_agregasi = {'Aman': 0.0, 'Waspada': 0.0, 'Bahaya': 0.0}
    for rule in rules_aktif:
        kategori = rule['output']
        if rule['alpha'] > hasil_agregasi[kategori]:
            hasil_agregasi[kategori] = rule['alpha']
    return hasil_agregasi



# 4. DEFUZZIFIKASI (SUGENO & MAMDANI)
SUGENO_KONSTANTA = {'Aman': 25.0, 'Waspada': 50.0, 'Bahaya': 85.0}

def defuzzifikasi_sugeno(hasil_agregasi):
    if not isinstance(hasil_agregasi, dict):
        raise ValueError("Input harus berupa dict")
    if any(v < 0 or v > 1 for v in hasil_agregasi.values()):
        raise ValueError("Nilai alpha harus antara 0 dan 1")

    pembilang = 0.0   
    penyebut  = 0.0   
    for kategori, alpha in hasil_agregasi.items():
        pembilang += alpha * SUGENO_KONSTANTA[kategori]
        penyebut  += alpha
    if penyebut == 0:
        return 0.0, "Tidak Terdeteksi"
    skor = round(pembilang / penyebut, 4)
    return skor, _label_dari_skor(skor)

MAMDANI_OUTPUT_MF = {
    'Aman':    (0,  25,  50),
    'Waspada': (25, 50,  75),
    'Bahaya':  (50, 75, 100)
}

def defuzzifikasi_mamdani(hasil_agregasi, resolusi=1000):
    x_vals = np.linspace(0, 100, resolusi + 1)
    mu_gabungan = np.zeros(resolusi + 1)
    for kategori, alpha in hasil_agregasi.items():
        a, b, c = MAMDANI_OUTPUT_MF[kategori]
        mu = np.array([fungsi_segitiga(x, a, b, c) for x in x_vals])
        mu_clip = np.minimum(mu, alpha)
        mu_gabungan = np.maximum(mu_gabungan, mu_clip)
    penyebut = np.sum(mu_gabungan)
    if penyebut == 0:
        return 0.0, "Tidak Terdeteksi"
    skor = round(float(np.sum(x_vals * mu_gabungan) / penyebut), 4)
    return skor, _label_dari_skor(skor)

def _label_dari_skor(skor):
    if skor < 40:
        return "Aman"
    elif skor < 65:
        return "Waspada"
    else:
        return "Bahaya"


# 5. TEST BLOK UTAMA
if __name__ == "__main__":
    gambar_test = "Img_test/tomato_healthy.png" 
    
    if not os.path.exists(gambar_test):
        print(f"File gambar tidak ditemukan di path: {gambar_test}")
    else:
        print("=== TAHAP 1: PREDIKSI CNN ===")
        skor_cnn, kelas_terdeteksi, akurasi = dapatkan_skor_penyakit_cnn(gambar_test)
        print(f"Prediksi Penyakit : {kelas_terdeteksi}")
        print(f"Confidence Model  : {akurasi:.2f}%")
        print(f"Skor untuk Fuzzy  : {skor_cnn:.2f}%")

        # Inisialisasi Data Lingkungan
        prediksi_cnn_confidence = skor_cnn 
        suhu_saat_ini = 35
        kelembapan_tanah = 10
        curah_hujan = 5
        umur_hari = 20

        print("\n=== TAHAP 2: FUZZIFIKASI ===")
        f_penyakit = fuzzifikasi_penyakit(prediksi_cnn_confidence)
        f_suhu = fuzzifikasi_suhu(suhu_saat_ini)
        f_kelembapan = fuzzifikasi_kelembapan(kelembapan_tanah)
        f_hujan = fuzzifikasi_hujan(curah_hujan)
        f_umur = fuzzifikasi_umur(umur_hari)
     
        print("Skor Penyakit :", f_penyakit)
        print("Suhu          :", f_suhu)
        print("Kelembapan    :", f_kelembapan)
        print("Curah Hujan   :", f_hujan)
        print("Umur Tanaman  :", f_umur)
        
        print("\n=== TAHAP 3: INFERENSI ===")
        evaluasi_rules = inferensi(f_penyakit, f_suhu, f_kelembapan, f_hujan, f_umur)
        hasil_akhir_inferensi = agregasi_aturan(evaluasi_rules)
        print("Hasil Agregasi :", hasil_akhir_inferensi)
     
        print("\n=== TAHAP 4: DEFUZZIFIKASI ===")
        skor_sugeno, label_sugeno = defuzzifikasi_sugeno(hasil_akhir_inferensi)
        print("--- Hasil Sugeno ---")
        print(f"Skor Risiko : {skor_sugeno}")
        print(f"Status      : {label_sugeno}")
     
        skor_mamdani, label_mamdani = defuzzifikasi_mamdani(hasil_akhir_inferensi)
        print("\n--- Hasil Mamdani ---")
        print(f"Skor Risiko : {skor_mamdani}")
        print(f"Status      : {label_mamdani}")
        print("=============================")