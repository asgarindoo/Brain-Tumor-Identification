# Brain Tumor Identification

## Deskripsi
Kode ini bertujuan untuk memuat data citra tumor otak dari dua folder yang berbeda ("yes" dan "no"), melakukan proses pra-pemrosesan, termasuk resize, segmentasi, dan ekstraksi fitur, dan akhirnya

## Persyaratan
1. Python 3.x
2. Library OpenCV (cv2)
3. Library NumPy (numpy)
4. Library pandas (pandas)
4. Folder yang berisi dataset citra tumor otak dengan dua subfolder: "yes" (berisi citra dengan tumor) dan "no" (berisi citra tanpa tumor)

## Dataset
https://www.kaggle.com/code/happygerypangestu/brain-tumor-glcm-classification/input

## Cara instalasi library yang dibutuhkan
pip install opencv-python numpy pandas

## Penggunaan
1. Tentukan path ke folder "no" dan "yes" di variabel tumor_no dan tumor_yes.
2. Jalankan fungsi load_data_with_resize untuk memuat data dari folder "no" dan "yes" dengan resize.
3. Gabungkan data dari kedua folder.
4. Jalankan fungsi process_image_with_resize untuk melakukan resize, segmentasi, dan ekstraksi fitur pada citra.

## Struktur Folder
- Folder `brain_tumor_dataset` harus berisi dua subfolder: "no" (berisi citra tanpa tumor) dan "yes" (berisi citra dengan tumor).
- Pastikan struktur folder sesuai dengan yang diharapkan oleh kode.

## Fungsi Utama
1. `load_data_with_resize(folder, label, target_size=(300, 300))`: Memuat data dari folder dengan resize citra ke ukuran tertentu.
2. `process_image_with_resize(img)`: Melakukan resize, segmentasi, dan ekstraksi fitur pada citra.
3. `erode_image(img, kernel, iterations)`: Melakukan operasi morfologi erosi menggunakan OpenCV.
4. `manual_erode(img, kernel, iterations=1)`: Melakukan operasi morfologi erosi secara manual.
5. `extract_features(img)`: Melakukan ekstraksi fitur manual dari citra.

## Catatan
Pastikan untuk memastikan bahwa semua path file sesuai dengan lokasi sebenarnya di sistem Anda sebelum menjalankan kode.

"""
