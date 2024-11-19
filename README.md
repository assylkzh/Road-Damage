# Automatic Classification of Road Damage

This project utilizes a deep learning model based on an enhanced version of ResNet50 to classify road conditions into four categories:
 "Good" , "Poor" , "Satisfactory" , "Very Poor" 
 The project demonstrates the implementation of a robust image classification pipeline, including data preprocessing, training, evaluation, and performance comparison with a baseline model.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation and Requirements](#installation-and-requirements)
5. [Usage](#usage)
6. [Results](#results)
7. [References](#references)
8. [Contributors](#сontributors)

---

## **Project Overview**
The aim of the “Automatic Classification of Road Damage” project is to classify images of road damages into four categories based on the extent of the damage. The categories are:

Good: No visible damage.

Poor: Noticeable damage, but not severe.

Satisfactory: Moderate damage, requiring attention.

Very Poor: Severe damage, very likely to be hazardous.

The dataset used is the "Road Damage Classification and Assessment" dataset from Kaggle platform, which contains images of roads with different levels of damage. The baseline model used is - ResNet50

The goal of this project is to classify road images into one of four conditions using a pretrained ResNet50 model fine-tuned with enhanced layers. The project involves:
- Data preprocessing and augmentation.
- Training and validating a baseline ResNet50 model.
- Developing an enhanced ResNet50 model with additional layers (Batch Normalization, Dropout, etc.).
- Comparing the performance of the two models.

---

## **Dataset**
The dataset contains road condition images, categorized into:
- **Good**
- **Poor**
- **Satisfactory**
- **Very Poor**

### **Folder Structure** (Before Splitting)
```
data/
├── good/
├── poor/
├── satisfactory/
├── very_poor/
```

The dataset is split into **train**, **validation**, and **test** sets:
- **70% Training**
- **15% Validation**
- **15% Testing**

---

## **Model Architecture**
### **Baseline Model**
- Pretrained ResNet50. 
- Fully connected layer replaced with a single `nn.Linear` layer for classification.

ResNet50 is a powerful convolutional neural network (CNN) architecture which was  pre-trained on the ImageNet dataset. The dataset contains millions of labeled images across 1,000 classes. ResNet50 has a deep network with 50 layers, and introduces residual connections (input of a layer is added to its output), and mainly, ResNet50 is designed for multi-class classification tasks.

### **Enhanced Model**
- Pretrained ResNet50 with frozen base layers.
- Modified final layers:
  - Fully connected layer with **512 units**
    - ReLU Activation
    - Batch Normalization
    - Dropout (with 40% probability)
  - Final classification layer with `NUM_CLASSES` outputs.

---

## **Installation and Requirements**

### **Dependencies**
- Python 3.8+ 
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn

### **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/username/road-condition-classification.git
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the dataset:
   - Place the dataset in the specified folder structure.
   - Update the `DATASET_PATH` variable in the script.

---

## **Usage**


### **1.  Run the Notebook**
Launch the project notebook:
```bash
jupyter notebook road_damage_classification.ipynb
```
Run cells one-by-one. 

### **2. Steps **
Data Preparation: Run cells to load and split the dataset.
Training: Train the baseline and enhanced ResNet50 model.
Evaluation: Evaluate the model on the test dataset.
Run cells to generate a classification report.
Run cells to display a confusion matrix heatmap.

---

## **Results**

### **Baseline Model**
| Metric         | Value      |
|----------------|------------|
| Accuracy       | **87%**    |
| Macro F1-Score | **83%**    |
| Weighted F1-Score | **86%** |

### **Enhanced Model**
| Metric         | Value      |
|----------------|------------|
| Accuracy       | **95%**    |
| Macro F1-Score | **95%**    |
| Weighted F1-Score | **95%** |

#### **Performance Improvements**
- The enhanced model's accuracy of 95% is a notable improvement over the baseline's 87%.
- "Poor" class’ Precision improves by 14% in  enhanced model
- In "Satisfactory” class recall improves from 60% to 90%, which indicates a reduction in misclassification.
- For “Very Poor" class Precision improves by 25%, and F1-score reaches 99%.


---

## **References **
This project uses the ResNet50 model from [PyTorch's torchvision library](https://pytorch.org/vision/stable/models.html). The dataset was sourced from Kaggle (https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment )

---
## **Contributors **
Assyl Kazhiakhmetova