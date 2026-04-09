---
title: "Two-Stage Hybrid Change Detection: Unsupervised Filtering with Deep Semantic Segmentation for Land Cover Monitoring"
authors:
  - Amogh Annigeri
  - Mahesh Khatawate
  - Raveesh P Nayak
  - Sathvik Kemtur
  - Dr. Sumaiya Pathan
affiliations:
  - KLE Technological University, Hubballi, India
---

# Two-Stage Hybrid Change Detection: Unsupervised Filtering with Deep Semantic Segmentation for Land Cover Monitoring

This repository contains the implementation of a novel two-stage hybrid pipeline for land-cover change detection using multi-temporal satellite imagery. The approach combines unsupervised clustering-based filtering with supervised deep semantic segmentation to achieve both computational efficiency and high semantic accuracy, particularly in resource-constrained operational environments.

## Abstract

Land-cover change detection is crucial for applications like disaster assessment, agricultural monitoring, and deforestation tracking. This work introduces a unified hybrid pipeline that integrates unsupervised clustering-based filtering with supervised VGG19-based semantic segmentation on the SECOND dataset. The first stage uses a VGG19 batch-normalized backbone as a feature extractor, applying K-Means clustering on feature differences to identify and filter samples with significant change. In the second stage, a modified VGG19 U-Net with early fusion is trained on this filtered subset to predict 37 semantic change classes. The model achieves a binary mean Intersection over Union (mIoU) of 64.68%, outperforming established baselines while maintaining architectural simplicity.

## Key Features

- **Two-Stage Hybrid Pipeline**: Combines unsupervised filtering and supervised semantic segmentation for efficient and accurate change detection.
- **Unsupervised Filtering (Stage 1)**:
  - Utilizes a VGG19 batch-normalized backbone as a feature extractor for temporal image pairs.
  - K-Means clustering is applied to high-dimensional feature differences to identify samples with significant change.
  - Filters out low-information samples, retaining 99.6% of training and 99.8% of test images, reducing computational load for the supervised stage.
- **Supervised Semantic Segmentation (Stage 2)**:
  - A modified VGG19 U-Net with early fusion is trained on the filtered subset.
  - Early fusion (6-channel input) allows the model to learn joint spatial and temporal features directly.
  - Predicts 37 semantic change classes (1 no-change + 36 directed transitions).
  - Employs weighted cross-entropy loss to address class imbalance, particularly for rare change classes.
  - AdamW optimizer with decoupled weight decay for effective regularization.
- **Dataset**: Evaluated on the high-resolution SECOND dataset, comprising 512x512 pixel temporal image pairs with pixel-level semantic annotations across six land cover classes.
- **Efficiency**: Achieves competitive performance with a simple VGG19 U-Net architecture, suitable for resource-constrained environments.

## Performance

The proposed model demonstrates strong performance on the SECOND dataset:

- **Binary mIoU**: 64.68%
- **Improvement over Baseline**: 5.38 percentage points over the FC-EF baseline (59.3%).
- **IoU for No-Change Class**: 82.05%
- **IoU for Change Class**: 47.31%
- **Training Efficiency**: Achieves convergence within 20 epochs on a single GPU.

## Methodology Overview

The pipeline operates in two main stages:

1.  **Unsupervised Filtering**: Raw bi-temporal image pairs are processed by a pretrained VGG19 backbone to extract deep convolutional features. Absolute feature differences are computed, and MiniBatch K-Means clustering (k=2) is applied to separate "change" and "no-change" regions. Image pairs are retained for further processing only if the proportion of detected change pixels exceeds a configurable threshold (default: 0.5%).

2.  **Supervised Classification**: The filtered image pairs are then fed into a modified VGG19 U-Net segmentation model. This model uses early fusion by concatenating the two temporal images into a 6-channel input. It is trained with weighted cross-entropy loss and AdamW optimizer to predict 37 semantic change classes.

During evaluation, samples that pass the unsupervised filter are processed by the segmentation network, while filtered-out samples are automatically assigned "no-change" predictions.

## Installation

_(Please note: Specific code for this project is not provided in the context. The following are general instructions based on the paper's description.)_

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/your_repository_name.git
    cd your_repository_name
    ```
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```
3.  **Install dependencies**:
    The project likely requires PyTorch, torchvision, scikit-learn, numpy, and other common machine learning libraries.

    ```bash
    pip install torch torchvision scikit-learn numpy pandas matplotlib
    ```

    Ensure you have a CUDA-compatible GPU and the appropriate PyTorch version installed for GPU acceleration.

4.  **Download the SECOND Dataset**:
    The paper mentions the use of the SECOND dataset. You will need to download and preprocess this dataset according to the instructions provided by its creators. Ensure the directory structure matches what the code expects (e.g., `SECOND_train_set`, `SECOND_total_test`).

## Usage

_(Please note: Specific code for this project is not provided in the context. The following are general instructions based on the paper's description.)_

### Training

To train the model, you would typically run a Python script, specifying parameters like learning rate, batch size, epochs, and paths to the dataset.

```bash
# Example command (actual command may vary)
python train.py --epochs 20 --batch_size 4 --lr 1e-4 --change_weight 5.0 --data_root /path/to/SECOND_dataset
```

### Evaluation

After training, you can evaluate the model's performance on the test set.

```bash
# Example command (actual command may vary)
python evaluate.py --model_path /path/to/best_pipeline_model.pt --data_root /path/to/SECOND_dataset
```

## References

The full list of references can be found in the original paper. Key references include:

- [1] Y. Chen, Z. Shi, P. Gong, et al., "A Self-Supervised Deep Learning Framework for Change Detection in Remote Sensing Images," IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022.
- [2] H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection," Remote Sensing, vol. 12, no. 10, pp. 1662, May 2020.
- [3] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," Proceedings of the International Conference on Learning Representations (ICLR), 2019.
- [4] R. C. Daudt, B. Le Saux, and A. Boulch, "Fully Convolutional Siamese Networks for Change Detection," Proceedings of the IEEE International Conference on Image Processing (ICIP), pp. 4063-4067, 2018.
- [5] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234-241, 2015.
- [6] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv preprint arXiv:1409.1556, 2014.
- [7] T. Celik, "Unsupervised Change Detection in Satellite Images Using Principal Component Analysis and k-Means Clustering," IEEE Geoscience and Remote Sensing Letters, vol. 6, no. 4, pp. 772-776, Oct. 2009.
