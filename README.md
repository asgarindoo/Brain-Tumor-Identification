# Brain Tumor Detection Using Image Processing

## Description

This project aims to detect brain tumors from MRI images using image processing techniques. The implementation involves resizing the images, applying manual thresholding, morphological operations (erosion), contour detection, feature extraction, and classification based on the extracted features.

## Requirements

1. Python 3.x
2. OpenCV library (cv2)
3. NumPy library (numpy)
4. Pandas library (pandas)
5. A folder containing the brain tumor image dataset with two subfolders: "yes" (contains images with tumors) and "no" (contains images without tumors)

# Dataset

[Brain Tumor Dataset](https://www.kaggle.com/code/happygerypangestu/brain-tumor-glcm-classification/input)

## Dataset Folder Structure

- The `brain_tumor_dataset` folder should contain two subfolders: "no" (contains images without tumors) and "yes" (contains images with tumors).

```
brain_tumor_dataset/
│
├── no/
│ ├── N1.jpg
│ ├── N2.jpg
│ └── ...
└── yes/
├── Y1.jpg
├── Y2.jpg
└── ...
```

# Installation

## Clone this repository

```sh
git clone https://github.com/asgarindoo/Brain-Tumor-Identification.git
```

## Navigate to the project directory

```sh
cd brain-tumor-detection
```

## Install the required packages

```sh
pip install opencv-python numpy pandas
```

# Usage

## Run the Script

```sh
python brain_tumor_detection.py
```

## Main Functions

1. `load_data_with_resize(folder, label, target_size=(300, 300))`: Loads data from the folder and resizes the images to a specified size.
2. `process_image_with_resize(img)`: Performs resizing, segmentation, and feature extraction on an image.
3. `manual_erode(img, kernel, iterations=1)`: Performs manual morphological erosion on an image.
4. `find_contours_manual(binary_img)`: Manually finds contours from a binary image.
5. `extract_features(img)`: Manually extracts features from an image.

## Notes

Ensure that all file paths match the actual locations on your system before running the code.
