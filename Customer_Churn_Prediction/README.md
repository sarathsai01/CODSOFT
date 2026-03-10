# Customer Churn Prediction

## Overview
Customer churn prediction is an important business problem where companies try to identify customers who are likely to stop using their services. By predicting churn, businesses can take preventive actions to retain customers.

This project builds a machine learning model to predict whether a bank customer will leave the service.

## Dataset
The dataset contains customer demographic and financial information used to predict churn behavior.

Dataset Source:
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

Target Variable:
- `Exited`
  - 0 → Customer stays
  - 1 → Customer leaves

Features include:
- Credit Score
- Geography
- Gender
- Age
- Balance
- Number of Products
- Active Membership
- Estimated Salary

## Methodology
The following workflow was followed:

1. Data preprocessing and feature selection
2. Encoding categorical variables
3. Splitting the dataset into training and testing sets
4. Training a Random Forest classification model
5. Evaluating the model using accuracy and confusion matrix

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Machine Learning

## Model Performance
The trained model achieved approximately **86% accuracy** in predicting customer churn.

## Conclusion
This project demonstrates how machine learning can help businesses analyze customer data and identify potential churn patterns to improve customer retention strategies.

## Internship Information
This project was completed during the **Machine Learning Internship at CodSoft**.
