🐱🐶 Image Classification – Cat vs Dog

This project explores binary image classification using Convolutional Neural Networks (CNNs) and state-of-the-art (SOTA) models. The task: given an image, predict whether it contains a cat or a dog.

The work includes building multiple custom CNNs, experimenting with hyperparameters, and comparing them with pretrained models such as ResNet50 and EfficientNet.

📂 Contents

Introduction

Dataset

Preprocessing & Augmentation

Custom Models

Model 1: Baseline

Model 2: + Dropout

Model 3: Deeper CNN + Dropout(0.3)

Model 4: Higher LR (failed)

Model 5: Best Custom Model

Transfer Learning (SOTA)

ResNet50

EfficientNet

Comparison of All Models

Grid Search Experiment

Results

📖 Introduction

This project applies TensorFlow/Keras for training CNNs on cat/dog datasets.

5 custom CNNs (sequential models) were built incrementally.

2 SOTA pretrained models (ResNet50 & EfficientNet) were tested.

Evaluation metrics: Accuracy, Precision, Recall, F1-score, Loss.

📊 Dataset
Kaggle Dataset

24,998 images (12,499 cats, 12,499 dogs)

Train: 80% | Validation: 10% | Test: 10%

Custom Dataset

40 random Google images (20 cats, 20 dogs)

Train: 70% | Validation: 20% | Test: 10%

🔧 Preprocessing & Augmentation

Normalization: pixel values scaled to [0, 1]

Resizing: 224x224x3 (RGB)

Augmentations:

Brightness (0.8–1.2)

Shear

Zoom (±30%)

Width & Height shifts (±20%)

Horizontal flip

🧪 Custom Models
✅ Model 1 – Baseline CNN

Accuracy (Val): 75.3% | F1: 75.7%

✅ Model 2 – + Dropout(0.5)

Accuracy (Val): 74.6% | F1: 74.3%

✅ Model 3 – Deeper CNN + Dropout(0.3)

Accuracy (Val): 76.3% | F1: 74.9%

❌ Model 4 – Higher LR (0.01)

Accuracy stuck at ~50%, failed to converge.

🏆 Model 5 – Best Custom Model (20 epochs)

Accuracy (Val): 84.2% | F1: 84.6%

Trained further on custom dataset, achieving 100% accuracy on test images.

🚀 Transfer Learning (SOTA Models)
ResNet50 (frozen backbone)

Accuracy (Val): 89.6%

Recall: 100%, Precision lower (~82.8%) → tendency to over-predict dogs

EfficientNet (frozen backbone)

Accuracy (Val): 98.7%

Best overall performance, excellent generalization

📈 Comparison of All Models

Custom CNNs: gradually improved, best F1 ~84%

ResNet50: high training accuracy but some overfitting

EfficientNet: best overall → ~99% validation accuracy

🔍 Grid Search Experiment

Explored hyperparameters:

Optimizer: Adam / SGD

Learning rate: 0.001 / 0.0001

Activation: ReLU / LeakyReLU

Dropout: 0.2 / 0.3

Pooling: Max / Average

Evaluated via Validation F1-score

Best setup: SGD + ReLU + low LR (0.0001)

🏁 Results

Best Custom CNN (Model 5): 84.6% F1

Best Transfer Learning Model (EfficientNet): 98.7% accuracy

Takeaway: Transfer learning strongly outperforms small custom CNNs for this task.
