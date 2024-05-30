import cv2
import numpy as np
import os
import pandas as pd

# Tentukan path ke folder "no" dan "yes"
tumor_no = r"C:\Users\Asgarindo\Downloads\brain_tumor_dataset\no"
tumor_yes = r"C:\Users\Asgarindo\Downloads\brain_tumor_dataset\yes"

# Fungsi untuk memuat data dari folder dengan resize
def load_data_with_resize(folder, label, target_size=(300, 300)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize citra
            img_resized = cv2.resize(img, target_size)
            data.append((img_resized, label))
        else:
            print(f"Gagal membaca gambar: {img_path}")
    return data

# Memuat data dari folder "no" dan "yes" dengan resize
data_no = load_data_with_resize(tumor_no, 0)
data_yes = load_data_with_resize(tumor_yes, 1)

# Gabungkan data dari kedua folder
data = data_no + data_yes

# Fungsi untuk melakukan resize, segmentasi, dan ekstraksi fitur
def process_image_with_resize(img):
    global eroded_img
    
    # Resize citra
    target_size = (400, 400)
    img_resized = cv2.resize(img, target_size)
    
    # Thresholding manual pada gambar grayscale
    threshold_value = 128  # Contoh nilai threshold
    _, thresholded_img = cv2.threshold(img_resized, threshold_value, 255, cv2.THRESH_BINARY)

    # Terapkan operasi morfologi (Contoh: Erosi manual)
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    eroded_img = erode_image(thresholded_img, kernel, iterations=2)

    # Ekstraksi fitur dari citra yang telah diolah
    features = extract_features(eroded_img)

    return features

# Fungsi untuk melakukan operasi morfologi erosi menggunakan OpenCV
def erode_image(img, kernel, iterations):
    # Lakukan erosi pada citra menggunakan fungsi cv2.erode
    eroded_img = cv2.erode(img, kernel, iterations=iterations)
    return eroded_img

# Fungsi untuk melakukan operasi morfologi erosi manual
def manual_erode(img, kernel, iterations=1):
    # Mendapatkan dimensi citra
    height, width = img.shape
    
    # Mendapatkan dimensi kernel
    k_height, k_width = kernel.shape
    
    # Mendapatkan ukuran padding
    pad_height = k_height // 2
    pad_width = k_width // 2
    
    # Buat citra hasil erosi
    eroded_img = np.zeros((height, width), dtype=np.uint8)
    
    # Looping melalui jumlah iterasi
    for _ in range(iterations):
        # Looping melalui citra
        for i in range(pad_height, height - pad_height):
            for j in range(pad_width, width - pad_width):
                # Inisialisasi nilai minimum
                min_val = 255
                
                # Looping melalui kernel
                for m in range(k_height):
                    for n in range(k_width):
                        # Koordinat citra
                        x = i + m - pad_height
                        y = j + n - pad_width
                        
                        # Periksa apakah di dalam batas citra
                        if x >= 0 and x < height and y >= 0 and y < width:
                            # Peroleh nilai minimum
                            min_val = min(min_val, img[x, y])
                
                # Tetapkan nilai minimum sebagai nilai piksel erosi
                eroded_img[i, j] = min_val
        
        # Perbarui citra input dengan citra hasil erosi untuk iterasi selanjutnya
        img = eroded_img.copy()
    
    return eroded_img


# Fungsi untuk melakukan ekstraksi fitur manual dari citra
def extract_features(img):
    # Temukan kontur dari citra
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Buat daftar untuk menyimpan hasil ekstraksi fitur
    features = []

    # Iterasi melalui kontur dan ekstrak ciri-ciri
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Pastikan perimeter tidak nol untuk menghindari pembagian oleh nol
        if perimeter == 0:
            continue
        
        # Hitung metric (bulatan) dari kontur
        metric = 4 * np.pi * area / perimeter ** 2
        
        # Hitung eccentricity
        x, y, w, h = cv2.boundingRect(contour)
        if w > h:
            w, h = h, w
        if h != 0:
            eccentricity = np.sqrt(1 - (w * w) / (h * h))
        else:
            eccentricity = np.nan

        # Tambahkan hasil ekstraksi ke daftar
        features.append([area, perimeter, metric, eccentricity])

    return features

# Memproses semua gambar dalam dataset
all_features = []

for img, label in data:
    features = process_image_with_resize(img)
    if features:
        for feature in features:
            feature.append(label)
        all_features.extend(features)

# Buat DataFrame dari semua fitur yang diekstraksi
df = pd.DataFrame(all_features, columns=['Luas', 'Keliling', 'Metric', 'Eccentricity', 'Label'])

# Tampilkan tabel
print(df)
print("====================================================")

# Contoh pemrosesan satu gambar dari data
image_path = os.path.join(tumor_yes, 'Y7.jpg')  # Ubah path sesuai lokasi sebenarnya
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Proses gambar dan ekstrak fitur
    features = process_image_with_resize(img)

    if features:
        df = pd.DataFrame(features, columns=['Luas', 'Keliling', 'Metric', 'Eccentricity'])
        print(df)

        # Tampilkan citra-citra yang diproses (opsional)
        cv2.imshow('Original Image', img)
        cv2.imshow('Segmented Image', eroded_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print(f"Gagal membaca gambar: {image_path}. Pastikan path file benar dan gambar tersedia.")
