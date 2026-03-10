# Credit Card Fraud Detection

## Overview
This project focuses on building a machine learning model to detect fraudulent credit card transactions. Fraud detection is an important application of machine learning in the financial sector to help prevent financial losses and identify suspicious activities.

The goal of this project is to analyze transaction data and classify whether a transaction is fraudulent or legitimate using machine learning algorithms.

## Dataset
The dataset used for this project contains credit card transaction records with multiple features describing transaction details.

Dataset Source:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Files Used:
- fraudTrain.csv
- fraudTest.csv

Target Variable:
- `is_fraud`  
  - 0 → Legitimate transaction  
  - 1 → Fraudulent transaction

## Methodology
The following steps were performed in this project:

1. Data preprocessing and cleaning
2. Handling categorical variables using encoding
3. Splitting the dataset into training and testing data
4. Training a Random Forest classifier
5. Evaluating model performance using accuracy, precision, recall, and confusion matrix

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Machine Learning Algorithms

## Model Performance
The trained model achieved an accuracy of approximately **99.7%** on the test dataset while successfully identifying a significant number of fraudulent transactions.

## Conclusion
This project demonstrates how machine learning can be applied to financial transaction data to detect fraud effectively and support automated fraud detection systems.

## Internship Information
This project was completed as part of the **Machine Learning Internship at CodSoft**.
