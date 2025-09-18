# 🐱🐶 Image Classification – Cat vs Dog  

This project explores **binary image classification** using **Convolutional Neural Networks (CNNs)** and **state-of-the-art (SOTA) models**.  
The task: given an image, predict whether it contains a **cat** or a **dog**.  

The work includes building multiple custom CNNs, experimenting with hyperparameters, and comparing them with pretrained models such as **ResNet50** and **EfficientNet**.  

---

## 📂 Contents
- [Introduction](#-introduction)  
- [Dataset](#-dataset)  
- [Preprocessing & Augmentation](#-preprocessing--augmentation)  
- [Custom Models](#-custom-models)  
  - **Model 1**: Baseline  
  - **Model 2**: + Dropout  
  - **Model 3**: Deeper CNN + Dropout(0.3)  
  - **Model 4**: Higher LR (failed)  
  - **Model 5**: Best Custom Model  
- [Transfer Learning (SOTA Models)](#-transfer-learning-sota-models)  
  - **ResNet50**  
  - **EfficientNet**  
- [Comparison of All Models](#-comparison-of-all-models)  
- [Grid Search Experiment](#-grid-search-experiment)  
- [Results](#-results)  

---

## 📖 Introduction
This project applies **TensorFlow/Keras** for training CNNs on cat/dog datasets.  

- 5 **custom CNNs** (sequential models) were built incrementally.  
- 2 **SOTA pretrained models** (ResNet50 & EfficientNet) were tested.  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, Loss.  

---

## 📊 Dataset
**Kaggle Dataset**  
- **24,998 images** (12,499 cats, 12,499 dogs)  
- Train: 80% | Validation: 10% | Test: 10%  

**Custom Dataset**  
- **40 random Google images** (20 cats, 20 dogs)  
- Train: 70% | Validation: 20% | Test: 10%  

---

## 🔧 Preprocessing & Augmentation
- **Normalization**: pixel values scaled to `[0, 1]`  
- **Resizing**: `224x224x3` (RGB)  
- **Augmentations**:  
  - Brightness (0.8–1.2)  
  - Shear  
  - Zoom (±30%)  
  - Width & Height shifts (±20%)  
  - Horizontal flip  

---

## 🧪 Custom Models

### ✅ Model 1 – Baseline CNN  
- **Validation Accuracy**: 75.3%  
- **F1-score**: 75.7%  

---

### ✅ Model 2 – + Dropout(0.5)  
- **Validation Accuracy**: 74.6%  
- **F1-score**: 74.3%  

---

### ✅ Model 3 – Deeper CNN + Dropout(0.3)  
- **Validation Accuracy**: 76.3%  
- **F1-score**: 74.9%  

---

### ❌ Model 4 – Higher LR (0.01)  
- **Validation Accuracy**: ~50%  
- **F1-score**: 0%  
- → Model failed to converge.  

---

### 🏆 Model 5 – Best Custom Model (20 epochs)  
- **Validation Accuracy**: 84.2%  
- **F1-score**: 84.6%  
- **Test Performance**: Balanced, strong generalization.  
- Trained on **custom dataset** → achieved **100% accuracy** on test images.  

---

## 🚀 Transfer Learning (SOTA Models)

### ResNet50 (frozen backbone)  
- **Validation Accuracy**: 89.6%  
- **Precision**: 82.8%  
- **Recall**: 100%  
- → Great at catching positives, but over-predicts dogs.  

---

### EfficientNet (frozen backbone)  
- **Validation Accuracy**: 98.7%  
- **Very stable loss & accuracy**  
- → **Best model overall** in this project.  

---

## 📈 Comparison of All Models
- **Custom CNNs**: Steady improvements, Model 5 best (F1 ~84%).  
- **ResNet50**: Very strong but showed some overfitting.  
- **EfficientNet**: Best performance overall with ~99% validation accuracy.  

---

## 🔍 Grid Search Experiment
Hyperparameters tested:  

- **Optimizer**: Adam / SGD  
- **Learning rate**: 0.001 / 0.0001  
- **Activation**: ReLU / LeakyReLU  
- **Dropout**: 0.2 / 0.3  
- **Pooling**: Max / Average  

📊 **Best setup**:  
- Optimizer: **SGD**  
- LR: **0.0001**  
- Activation: **ReLU**  
- Pooling: **Max**  

---

## 🏁 Results
- **Best Custom CNN (Model 5)**: **84.6% F1**  
- **Best Transfer Learning Model (EfficientNet)**: **98.7% accuracy**  
- **Takeaway**: Transfer learning strongly outperforms small custom CNNs for this task.  
