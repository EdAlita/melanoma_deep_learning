# Melanoma Detection Using Deep Learning

## Table of Contents

1. [**Introduction**](#introduction)
2. [**Objectives**](#objectives)
3. [**Data Description**](#data-description)
4. [**Folders**](#folders)
6. [**Data Manipulation**](#data-manipulation)
7. [**Model Architecture**](#model-architecture)
8. [**Contributing**](#contributing)
9. [**Installation and Usage**](#installation-and-usage)
10. [**License**](#license)
11. [**Acknowledgments**](#acknowledgments)

## Introduction

The melanoma detection project is part of a larger initiative focused on Computer Aided Diagnosis (CADx). The main purpose of this project is to develop a CADx medical system that assists physicians in delivering diagnoses. Specifically, the project centers on skin analysis, with an emphasis on melanoma detection. This project involves implementing deep learning techniques to develop algorithms that can provide a second opinion in dermatological diagnoses, particularly in identifying and classifying melanoma from dermoscopic images.

## Objectives

The primary objectives of the project are as follows:

1. **Development of a CADx System**: Create a computer-aided diagnostic system for melanoma detection. This system aims to support healthcare professionals by providing reliable second opinions.
2. **Algorithm Development**: Develop and refine algorithms capable of diagnosing melanoma from dermoscopic images. These algorithms should be able to distinguish between nevi and malignant cancers.
3. **Deep Learning Approach**: Utilize deep learning techniques to enhance the accuracy and reliability of the diagnostic process. This involves a comprehensive review of current literature, followed by the application of advanced deep learning models.

## Data Description

The melanoma[^1] detection project utilizes a comprehensive dataset of dermoscopic images. This dataset is pivotal for the training and validation of deep learning models designed to distinguish between benign nevus[^2] and malignant melanoma.

The lesion images come from the HAM10000 Dataset (ViDIR Group,
Medical University of Vienna), the BCN_20000 Dataset (Hospital Clínic de Barcelona) and the MSK Dataset
(ISBI 2017), hence images were acquired with a variety of dermatoscope types and from different anatomic
sites. Images were acquired from a historical sample of patients that presented for skin cancer screening
from several different institutions.

![Sample of Dataset](figures/new_image_3.png)
*Figure 1: Example of the type of lesion in the dataset*

[^1]: a type of cancer that develops from the pigment-producing cells known as melanocytes.
[^2]: Nevus (pl.: nevi) is a nonspecific medical term for a visible, circumscribed, chronic lesion of the skin or mucosa

### Binary classification dataset

The binary problem of classifying Nevus images vs all the others. We will give you more than 15000 images, being half of them nevus and the other half a combination of abnormal areas to train the system. The test set will be open the last week of the project

![Binary Dataset](figures/Binary.png)
*Figure 2: Details of the binary datset*

- 15195 images for training (with ground-truth), approx 50% nevus & lesions
- 3796 images for validation (with ground-truth), approx 50% nevus & lesions
-  XXXX images for testing (without ground-truth), unknown distribution

### Multiclass classification dataset

A three-class problem consisting on the classification of cancers: melanoma vs basal cell carcinoma vs squamous cell carcinoma. The training set consists on more than 5000 images, being 50% melanoma, 40% basal cell carcinoma and only 10% squamous cell carcinoma (imbalanced problem).

![Multiclass Dataset](figures/Multi-Class.png)
*Figure 3: Details of the Multi-class datset*

- 5082 images for training (with gtruth), approx 50% mel / 40% bcc / 10% scc
- 1270 images for validation (with gtruth), 50% mel / 40% bcc / 10% scc
- XXXX images for testing (without gtruth), unknown distribution


## Folders
- [**Classifiers**](classifiers): Network architecture, trainning, val and test files.
- [**Noteebooks**](noteebooks): Notebooks of the implementation of the deep learning classifier.
- [**Evaluation**](evaluation): Metrics use to evaluated our deep learning implementation.
- [**Utils**](utils): helping functions.
- [**Literature**](literature): Journals use to based this implementation.
- [**Generators**](generators): GAN implementation.

## Data Manipulation

### Preprocessing
#### Algorithm Steps
##### Function Definition
The function `preprocess` is designed to take an image and apply Contrast Limited Adaptive Histogram Equalization (CLAHE) along with min-max normalization:

```python
def preprocess(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    B, G, R = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    B = clahe.apply(B)
    G = clahe.apply(G)
    R = clahe.apply(R)
    clahe_image = cv2.merge([B, G, R])
    min_max_normalized_image = cv2.normalize(clahe_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return min_max_normalized_image
```
##### Step 1: Color Channel Splitting and CLAHE
- **Splitting Channels**: The image is split into its Blue (B), Green (G), and Red (R) color channels.

- **CLAHE**: Each channel undergoes CLAHE, which enhances the contrast of the image. This step is essential for bringing out hidden features in the image, especially in areas with close-to-similar intensities.

##### Step 2: Merging and Normalization
- **Merging Channels**: After applying CLAHE, the channels are merged back to form a single image.

- **Normalization**: The merged image then undergoes min-max normalization. This step scales the pixel values to a standard range (0-255 in this case), which is a common practice in image processing for maintaining uniformity across different images.

### Masks Generation

#### Algorithm Steps

##### Step 1: Preprocessing the Image

First, the image undergoes preprocessing to enhance its contrast and color features.

```python
import cv2
import numpy as np

image = cv2.imread('../data/train/others/bkl00763.jpg')
clip_limit = 2.0
tile_grid_size = (8, 8)

B, R, G = cv2.split(image)
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
B = clahe.apply(B)
R = clahe.apply(R)
G = clahe.apply(G)
image = cv2.merge([B, R, G])
```

##### Step 2: Gaussian Filtering

The image is smoothed using a Gaussian filter to reduce noise.

```python
gaussian_filtered = cv2.GaussianBlur(image, (25, 25), 0)
```

##### Step 3: Conversion to LAB Color Space and Channel Thresholding

The Gaussian-filtered image is converted to the LAB color space, and the mean values of the L, A, and B channels are calculated to be used as thresholds.

```python
lab_image = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2Lab)
l_mean = np.mean(lab_image[:, :, 0])
a_mean = np.mean(lab_image[:, :, 1])
b_mean = np.mean(lab_image[:, :, 2])

_, l_binary = cv2.threshold(lab_image[:, :, 0], l_mean, 255, cv2.THRESH_BINARY)
_, a_binary = cv2.threshold(lab_image[:, :, 1], a_mean, 255, cv2.THRESH_BINARY)
_, b_binary = cv2.threshold(lab_image[:, :, 2], b_mean, 255, cv2.THRESH_BINARY)
```

##### Step 4: Intersection of Binary Channels

The binary channels are combined to create an intersection image, isolating the regions of interest.

```python
intersection = cv2.bitwise_and(b_binary, cv2.bitwise_and(a_binary, l_binary))
```

##### Step 5: Median Filtering

A median filter is applied to the intersection image to reduce noise and smooth edges.

```python
median_filtered = cv2.medianBlur(intersection, 15)
```

![Mask Generation Results](figures/maks_generation_algorithm.png)
*Figure 4: Details of the steps to generate the masks*

## Model Architecture
### Binary Classification

After a lot of testing of different models, our best approach was using a transfer learning of `inception_v3` with the `Inception_V3_Weights` from the moodle `torchvision.models` version 0.15.2. We use a last classification layers with linear and Relu activations. We use the maxpooling from all the features of the `inception_v3` to get 2048, then a linnear layer to reduce them to 1024 and at last one to reduce them from 512 to 2 classes.

![Binary Dataset work flow](figures/binary_class.png)
*Figure 5: Details of our Deep Learning layers*

In the next figure you can observe a detail result of our binary result test of different approach of the classifcation problem.

![Binary test results](figures/binary_results.png)
*Figure 6: Details of Binary testing scheme*

## Contributing
- [Yusuf B. Tanrıverdi](https://github.com/yusuftengriverdi)
- [Edwing Ulin](https://github.com/EdAlita)

## Installation and Usage

### Requirements
- Python along with additional dependencies listed in a `requirements.txt` file.

### Creating a Virtual Environment
To avoid conflicts with other Python projects, it's recommended to create a virtual environment:
1. Install `virtualenv` if you haven't already: `pip install virtualenv`
2. Create a new virtual environment: `virtualenv venv` (or `python -m venv venv` if using Python's built-in venv)
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
4. Your command prompt should now show the name of the activated environment.

### Installation
1. With the virtual environment activated, install the necessary Python packages: `pip install -r requirements.txt`

### Usage

In  order to get the new dataset to train this part, we need to build it from the original dataset. In order to accomplish this we created the next function.

```bash
python classifiers/organize.py --root_path "../data/val/others/" --target_folder "../data_mult/val/"
```
- `--root_path`: is the original folder to rearrange.
- `--target_folder`: is the destination folder of the new dataset

> Note only run this after creating the destination folder


### Fitting
This function are share with binary and multi. Only change the folder name and the dataset used.

To fit the model we created the next function that can be call in the next form:
```bash
python classifiers/binary_classifier/fit.py ----number_epochs 10 --batch_size 16 --train_dir data/train/ --save_dir out/run_10/
```
- `--batch_size`: number of batch to use on the load data part
- `--number_epochs`: number of epochs to run on the train
- `--train_dir`: train data directory.
- `--save_dir`: save data directory.

> Note, only run this after creating the destination folder and the learning rate will be roating from a set initial and final depending on the epoch the step is determine.

### Validation

To validate the model we created the next function that can be call in the next form:
```bash
python classifiers/multiple_classifier/val.py --mode best --val_dir data/val/ --root_out out/run_4/ --batch_size 16
```
- `--batch_size`: number of batch to use on the load data part.
- `--val_dir`: validation data directory.
- `--root_out`: folder of the save instances.
- `--train_dir`: train data directory.
- `--mode`: used the `best`

> Note, the `normal` mode is only used if you have instances from the last versions of fit.

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments
