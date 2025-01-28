# **Semi-Supervised Learning for Fashion MNIST with Variational Autoencoders and SVMs**

## **Research Objectives**
The objective of this study is to explore the potential of **semi-supervised learning** in leveraging **unlabeled data** to enhance classification performance. In many real-world applications, acquiring labeled datasets is both expensive and time-consuming. This research investigates whether **Variational Autoencoders (VAEs)** can effectively learn **latent representations** from both labeled and unlabeled data and whether these representations improve classification accuracy when used with **Support Vector Machines (SVMs).**

Specifically, this study aims to:
- Evaluate the ability of **deep generative models** (VAEs) to extract meaningful features from Fashion MNIST.
- Compare classification performance when varying the number of labeled training samples.
- Investigate the impact of **KL divergence scaling**, **batch normalization**, and **different activation functions** on model performance.
- Assess whether SVMs trained on latent representations can outperform traditional supervised approaches with limited labeled data.

This research contributes to **efficient learning in data-constrained environments** by demonstrating how **hybrid approaches combining deep learning and classical machine learning methods** can improve performance while minimizing reliance on labeled data.

---

## **Overview**
This project explores a **semi-supervised learning approach** using **Variational Autoencoders (VAEs)** and **Support Vector Machines (SVMs)** to classify images from the **Fashion MNIST** dataset. The VAE is used to **learn a compressed latent space representation** of the dataset, which is then utilized by an SVM classifier to make predictions.

### **Why is this important?**
- **Limited labeled data**: In many machine learning applications, labeled data is scarce and expensive to acquire.
- **Leveraging unlabeled data**: VAEs can learn useful representations from large amounts of unlabeled data to improve classification accuracy.
- **Hybrid approach**: Combining **deep generative models (VAEs)** with **traditional classifiers (SVMs)** provides a powerful framework for feature extraction and classification.

---

## **Dataset**
The project utilizes the **Fashion MNIST dataset**, which consists of **60,000 training** and **10,000 test images** of clothing items categorized into **10 classes**.

- The dataset was **split into labeled and unlabeled subsets** to train the VAE, with labeled sets containing **100, 600, 1000, and 3000 samples**.
- The **VAE learns a compressed representation of the images** using an encoder-decoder architecture.
- The **SVM is trained on the latent representations** of the labeled data to classify test samples.

---

## **Model Architecture**

### **Variational Autoencoder (VAE)**
- **Encoder**:
  - 2 hidden layers of **600 neurons** each
  - **Softplus activation** with **Batch Normalization**
  - Outputs **mean (μ) and log-variance (log σ²)** of the latent distribution
  - Latent space dimensionality was varied between **10 and 100**, with **50** being optimal.

- **Latent Space**:
  - A **latent dimension of 50** was used to capture meaningful features.

- **Decoder**:
  - Mirrors the encoder with **2 hidden layers of 600 neurons** each
  - **Softplus activation** followed by **Sigmoid activation**
  - Batch normalization was added to improve stability.

### **Loss Function**
The model is trained using a **combination of Binary Cross-Entropy (BCE) loss and KL Divergence**:
- **Binary Cross-Entropy (BCE)** ensures accurate image reconstruction.
- **KL Divergence** regularizes the latent space, ensuring meaningful representations.
- A **training scheduler gradually increases the weight of KL Divergence** to balance learning.

### **Classification Using SVM**
- The **VAE encoder extracts latent representations** of labeled samples.
- A **Support Vector Machine (SVM) with an RBF kernel** is trained on these latent features.
- The trained SVM is used to classify test samples based on their latent representations.

---

## **Experimental Results**
| **Number of Labeled Samples** | **Accuracy** |
|------------------------------|------------|
| 100                          | 69.85%     |
| 600                          | 77.20%     |
| 1000    
