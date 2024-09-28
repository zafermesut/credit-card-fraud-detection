# Credit Card Fraud Detection

This project demonstrates the use of **Autoencoders**, a deep learning-based unsupervised technique, to detect fraudulent credit card transactions. The model is trained to learn the patterns of non-fraudulent transactions and flag any deviations, which are likely to be fraudulent.

## Dataset

The dataset used for this project is sourced from Kaggle and contains information on credit card transactions labeled as fraudulent or non-fraudulent. The dataset can be downloaded from the following link:

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)

### Dataset Details:

- **Class 0**: Represents non-fraudulent transactions.
- **Class 1**: Represents fraudulent transactions.

## Project Overview

Fraud detection is a critical task in the financial domain, where identifying anomalies or suspicious transactions can help prevent significant financial loss. In this project, we employ an autoencoder model that is trained only on non-fraudulent transactions. Once trained, the model can reconstruct normal transactions with minimal loss, making it easier to detect outliers—fraudulent transactions.

The main steps involved in this project are:

1. Data Preprocessing (Normalization, Train-Test Split)
2. Autoencoder Model Architecture (Encoding & Decoding)
3. Model Training using non-fraudulent data
4. Fraud Detection based on reconstruction error
5. Performance evaluation using Logistic Regression and classification metrics

## Why Autoencoder?

Autoencoders are a powerful tool for anomaly detection, especially in cases where labeled data for fraudulent transactions is scarce or unbalanced. The model learns the intrinsic patterns of the normal (non-fraudulent) transactions during the encoding-decoding process. By learning to reconstruct the normal data, the model can easily detect transactions that deviate from this pattern, as they will have a higher reconstruction error.

In this project, the **reconstruction error** is used as a metric to distinguish between fraudulent and non-fraudulent transactions. Autoencoders are especially well-suited for:

- **Handling Imbalanced Data**: In fraud detection, fraudulent transactions are much fewer compared to normal ones, making traditional supervised models prone to overfitting on the majority class. Autoencoders mitigate this issue by focusing on learning the patterns of the majority class (non-fraudulent transactions).

## Model Architecture

The autoencoder in this project consists of:

- **Input Layer**: Takes the feature set from the transaction data.
- **Encoding Layers**: Compresses the input data into a smaller representation.
- **Decoding Layers**: Attempts to reconstruct the input data from the compressed representation.
- **Output Layer**: The final reconstructed output, which is compared with the original input to compute reconstruction error.

### Layers:

- Encoding: 100 units (tanh) -> 50 units (ReLU)
- Decoding: 50 units (tanh) -> 100 units (tanh)
- Output: Input size, activation (ReLU)

The model is trained using **Mean Squared Error (MSE)** as the loss function, and **Adadelta** as the optimizer.

## Typical Use Cases of Autoencoders

Autoencoders are commonly used in a variety of domains for anomaly detection and data compression. Some of the typical use cases include:

- **Fraud Detection**: As demonstrated in this project, autoencoders can detect abnormal patterns in financial transactions and network traffic data.
- **Outlier Detection**: Autoencoders are useful in detecting outliers in datasets, especially where anomalies are rare and data is mostly normal.
- **Image Denoising**: In computer vision, autoencoders are used to remove noise from images by learning the underlying clean image structure.
- **Dimensionality Reduction**: Autoencoders provide an alternative to traditional methods like PCA for reducing the dimensionality of data while preserving important features.
