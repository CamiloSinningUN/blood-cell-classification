# Blood Cell Classification with Transfer Learning and Advanced Data Augmentation

## Overview
This project was developed as part of the *Advanced Neural Networks and Deep Learning (AN2DL)* course at Politecnico di Milano. The objective was to classify eight different types of blood cells using deep learning techniques. We leveraged convolutional neural networks (ConvNeXt), transfer learning, and advanced data augmentation strategies to maximize accuracy.

## Dataset
The dataset consists of **13,759** RGB images (96x96 pixels) of eight types of blood cells. The distribution of classes is highly imbalanced, which necessitated careful preprocessing and augmentation.

![download](https://github.com/user-attachments/assets/1bd939d4-7474-4d15-8f1e-b9f9adecd7f3)

## Methodology

### Data Preprocessing
- **Outlier detection** using PCA and DBSCAN.
- **t-SNE validation** for duplicate removal.
- **Dataset balancing** with augmentation strategies.

### Model
- **Base Model**: ConvNeXtXLarge pretrained on ImageNet.
- **Fine-tuned layers**: Two fully connected layers with Swish activation.
- **Optimizations**:
  - Augmentations: RandAugment, GridMask, AugMix, ChannelShuffle.
  - Optimizers: AdamW, RMSprop, Lion (AdamW performed best).
  - Loss functions: Switched from CategoricalCrossEntropy to Focal Loss.

### Training
- **Environment**: Kaggle.
- **Best Model Achieved**:
  - **94% accuracy on Codabench**.
  - Fine-tuned ConvNeXtXLarge with 3 fully connected layers.
  - Advanced augmentation pipeline.

## Results
| Model | Local Val Accuracy | Train Time | Trainable Params | Codabench Accuracy |
|-------|--------------------|------------|------------------|--------------------|
| ConvNeXtXLarge (baseline) | 71.00% | 5s/epoch | 10M | 71.00% |
| ConvNeXtXLarge + Augmentations | 85.88% | 25s/epoch | 10.5M | 85.00% |
| **Final Model (Fine-tuned + Augmentations)** | **93.61%** | **85s/epoch** | **1.3B** | **94.00%** |

## Folders

This repository has the following folders, 

* **Baseline**: This consists of a model that uses a random guessing in order to solve the task
* **Hyperparameter search (hp-searching)**: This has the notebooks used for the bayesian search of the hyperparameters
* **Preprocessing**: This has the notebooks used for outlier detection and advance augmentation pipeline.
* **Trainning**: It has the notebooks used for trainning the best model (0.94 accuracy)
  
Most folders have an additional sub-folder that's called experiments which contains notebooks used for the intermediate models generated.

Additionally, there is the notebook model_comparison which describes synthetically some of the different network architectures that was explored.

Note: the Training Altantis notebook is just a template for Training Treasure Planet, it has no ooutput.

## Contributors
* [Matteo Panarotto](https://github.com/MatteoPana)
* [Camilo José Sinning López](https://github.com/CamiloSinningUN)
* [Oliver Stege](https://github.com/Smrevilo)
