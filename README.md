# Melanoma Detection Using Deep Learning

## Table of Contents

1. [**Introduction**](#introduction)
2. [**Objectives**](#objectives)
3. [**Folders**](#folders)
4. [**Installation and Usage**](#installation-and-usage)
5. [**Data Description**](#data-description)
6. [**Model Architecture**](#model-architecture)
7. [**Results**](#results)
8. [**Contributing**](#contributing)
9. [**License**](#license)
10. [**Acknowledgments**](#acknowledgments)

## Introduction

The melanoma detection project is part of a larger initiative focused on Computer Aided Diagnosis (CADx). The main purpose of this project is to develop a CADx medical system that assists physicians in delivering diagnoses. Specifically, the project centers on skin analysis, with an emphasis on melanoma detection. This project involves implementing deep learning techniques to develop algorithms that can provide a second opinion in dermatological diagnoses, particularly in identifying and classifying melanoma from dermoscopic images.

## Objectives

The primary objectives of the project are as follows:

1. **Development of a CADx System**: Create a computer-aided diagnostic system for melanoma detection. This system aims to support healthcare professionals by providing reliable second opinions.
2. **Algorithm Development**: Develop and refine algorithms capable of diagnosing melanoma from dermoscopic images. These algorithms should be able to distinguish between nevi and malignant cancers.
3. **Deep Learning Approach**: Utilize deep learning techniques to enhance the accuracy and reliability of the diagnostic process. This involves a comprehensive review of current literature, followed by the application of advanced deep learning models.

## Folders


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

## Data Description

The melanoma([^1]) detection project utilizes a comprehensive dataset of dermoscopic images. This dataset is pivotal for the training and validation of deep learning models designed to distinguish between benign nevus([^2]) and malignant melanoma.

The lesion images come from the HAM10000 Dataset (ViDIR Group,
Medical University of Vienna), the BCN_20000 Dataset (Hospital Clínic de Barcelona) and the MSK Dataset
(ISBI 2017), hence images were acquired with a variety of dermatoscope types and from different anatomic
sites. Images were acquired from a historical sample of patients that presented for skin cancer screening
from several different institutions.

![Sample of Dataset](figures/new_image_3.png)
*Figure 1: Example of the type of lesion in the dataset*

[^1] : a type of cancer that develops from the pigment-producing cells known as melanocytes.
[^2] : Nevus (pl.: nevi) is a nonspecific medical term for a visible, circumscribed, chronic lesion of the skin or mucosa

