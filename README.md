# jatj_football_transfermarkt

This repo is in essence, another version of JATJ's team approach to the model building our 2025 datascience course.  It summarize our Exploratory analysis of the dataset and how we decide to approach the modelling.  dtance.

# EDA


# Approaches
## Data Loading and Preparation: 
It starts by mounting Google Drive and loading several CSV files containing data about players, clubs, games, transfers, and more.

## Feature Engineering:

A comprehensive set of features is engineered by merging the different dataframes. These features include player attributes (age, height, position, foot), performance metrics (goals, assists, cards per 90 minutes, appearances), club performance (win rate, goals scored/conceded), transfer history (transfer count, fees), contract details, and flags for top leagues and foreign players. The target variable, market value, is also log-transformed to handle outliers.

## Data Splitting: The data is split into training and testing sets.

## Preprocessing Pipeline: 

A preprocessing pipeline is defined using ColumnTransformer to handle categorical features (one-hot encoding) and numerical features (standard scaling).

## Model Building and Evaluation:
Linear Regression: A Linear Regression model is built and evaluated using MAE, RMSE, and R². The notebook also visualizes the actual vs. predicted values, residuals, and feature coefficients.
Random Forest: A Random Forest Regressor model is built and evaluated using the same metrics. Feature importances are also visualized.
XGBoost: An XGBoost Regressor model is built and evaluated. Feature importances are visualized as well.
Deep Neural Network (DNN): Two versions of a Sequential Keras model are built with different architectures and training parameters (including Batch Normalization, Dropout, L2 regularization, and learning rate reduction). Both DNNs are trained and evaluated. Training history (loss and MAE), actual vs. predicted values, residuals, and error distributions are visualized for the DNN models.
