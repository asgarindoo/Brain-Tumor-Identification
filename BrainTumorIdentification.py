import cv2
import numpy as np
import os
import pandas as pd

# Tentukan path ke folder "no" dan "yes"
tumor_no = r"C:\Users\Asgarindo\Downloads\brain_tumor_dataset\no"
tumor_yes = r"C:\Users\Asgarindo\Downloads\brain_tumor_dataset\yes"

# Fungsi untuk memuat data dari folder dengan resize
def load_data_with_resize(folder, target_size=(300, 300)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize citra
            img_resized = cv2.resize(img, target_size)
            data.append((img_resized))
        else:
            print(f"Gagal membaca gambar: {img_path}")
    return data

# Memuat data dari folder "no" dan "yes" dengan resize
data_no = load_data_with_resize(tumor_no)
data_yes = load_data_with_resize(tumor_yes)

# Gabungkan data dari kedua folder
data = data_no + data_yes

# Fungsi untuk melakukan resize, segmentasi, dan ekstraksi fitur
def process_image_with_resize(img):
    global eroded_img, thresholded_img
    
    # Resize citra
    target_size = (400, 400)
    img_resized = cv2.resize(img, target_size)
    
    # Thresholding manual pada gambar grayscale
    threshold_value = 128  # Contoh nilai threshold
    thresholded_img = np.zeros_like(img_resized)
    thresholded_img[img_resized > threshold_value] = 255

    # Terapkan operasi morfologi (Contoh: Erosi manual)
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    eroded_img = manual_erode(thresholded_img, kernel, iterations=2)

    # Ekstraksi fitur dari citra yang telah diolah
    features = extract_features(eroded_img)

    return features

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

# Fungsi untuk menemukan kontur secara manual dari citra biner
def find_contours_manual(binary_img):
    contours = []
    visited = np.zeros_like(binary_img, dtype=bool)
    height, width = binary_img.shape

    # Fungsi untuk memeriksa apakah titik (x, y) valid dan belum dikunjungi
    def is_valid(x, y):
        return 0 <= x < width and 0 <= y < height and binary_img[y, x] != 0 and not visited[y, x]

    # Pergerakan 8 arah (atas, bawah, kiri, kanan, dan diagonal)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(height):
        for x in range(width):
            if is_valid(x, y):
                # Mulai pelacakan kontur dari titik yang valid
                contour = []
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if is_valid(cx, cy):
                        visited[cy, cx] = True
                        contour.append((cx, cy))
                        # Tambahkan tetangga yang valid ke stack
                        for dx, dy in directions:
                            nx, ny = cx + dx, cy + dy
                            if is_valid(nx, ny):
                                stack.append((nx, ny))
                if len(contour) > 0:
                    contours.append(np.array(contour))

    return contours

# Fungsi untuk melakukan ekstraksi fitur manual dari citra
def extract_features(img):
    # Temukan kontur dari citra secara manual
    contours = find_contours_manual(img)

    # Buat daftar untuk menyimpan hasil ekstraksi fitur
    features = []

    # Iterasi melalui kontur dan ekstrak ciri-ciri
    for contour in contours:
        if len(contour) < 3:  # Minimum kontur harus memiliki setidaknya 3 titik
            continue
        
        # Konversi kontur ke format yang dapat digunakan oleh cv2 untuk perhitungan area dan perimeter
        contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
        
        # Hitung area dan perimeter dari kontur
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

def classify_tumor(features):
    tumor_classification = 'Normal'
    
    # Thresholds untuk klasifikasi tumor
    jinak_thresholds = {
        'Luas': 500,  
        'Keliling': 500, 
        'Metric': 1.5, 
        'Eccentricity': 1.5 
    }
    
    ganas_thresholds = {
        'Luas': 1000,  
        'Keliling': 1000, 
        'Metric': 2.5, 
        'Eccentricity': 2.0 
    }

    for feature_values in features:
        if (feature_values[0] > ganas_thresholds['Luas'] or
            feature_values[1] > ganas_thresholds['Keliling'] or
            feature_values[2] > ganas_thresholds['Metric'] or
            feature_values[3] > ganas_thresholds['Eccentricity']):
            tumor_classification = 'Tumor Ganas'
            break
        elif (feature_values[0] > jinak_thresholds['Luas'] or
              feature_values[1] > jinak_thresholds['Keliling'] or
              feature_values[2] > jinak_thresholds['Metric'] or
              feature_values[3] > jinak_thresholds['Eccentricity']):
            tumor_classification = 'Tumor Jinak'
            break
            
    return tumor_classification

# Contoh pemrosesan satu gambar dari data
image_path = os.path.join(tumor_yes, 'Y6.jpg')  
# image_path = os.path.join(tumor_no, 'N21.jpg')  
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Proses gambar dan ekstrak fitur=
    features = process_image_with_resize(img)

    if features:
        print("====================================================")
        df = pd.DataFrame(features, columns=['Luas', 'Keliling', 'Metric', 'Eccentricity'])
        target = classify_tumor(features)
        
        df['target'] = target
        
        print(df)
        print("====================================================")
        
        # Tampilkan citra-citra yang diproses (opsional)
        cv2.imshow('Original Image', img)
        # Menampilkan citra thresholded
        cv2.imshow('Thresholded Image', thresholded_img)
        cv2.imshow('Morfologi Eroded Image', eroded_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print(f"Gagal membaca gambar: {image_path}. Pastikan path file benar dan gambar tersedia.")
