# jatj_football_transfermarkt

This repo is in essence, another version of JATJ's team approach to the model building our 2025 datascience course.  It summarize our Exploratory analysis of the dataset and how we decide to approach the modelling.  dtance.

# EDA

## Basic analysis - headcount
we start with understanding the dataset of players. as the result 

![download](https://github.com/user-attachments/assets/b433467f-a863-4c87-8e7c-2bead0edcfdd)

## Market value


# Approaches
## Data Loading and Preparation: 
It starts by mounting Google Drive and loading several CSV files containing data about players, clubs, games, transfers, and more.

## Feature Engineering:

A comprehensive set of features is engineered by merging the different dataframes. These features include player attributes (age, height, position, foot), performance metrics (goals, assists, cards per 90 minutes, appearances), club performance (win rate, goals scored/conceded), transfer history (transfer count, fees), contract details, and flags for top leagues and foreign players. The target variable, market value, is also log-transformed to handle outliers.

## Data Splitting: The data is split into training and testing sets. 

## Preprocessing Pipeline: 

A preprocessing pipeline is defined using ColumnTransformer to handle categorical features (one-hot encoding) and numerical features (standard scaling).


## Model Building and Evaluation:
### Linear Regression: 
A Linear Regression model is built and evaluated using MAE, RMSE, and R². The notebook also visualizes the actual vs. predicted values, residuals, and feature coefficients.

### Random Forest: 
A Random Forest Regressor model is built and evaluated using the same metrics. Feature importances are also visualized.



The evaluation metrics for the Random Forest model indicate a decent overall performance in predicting football player market values. 
  With a Mean Absolute Error (MAE) of approximately 0.476, the model’s predictions deviate only moderately from actual values on average, suggesting consistent accuracy. 
  The Root Mean Squared Error (RMSE) of 0.636 reflects a relatively low level of prediction error magnitude, reinforcing the model's stability. Most notably, 
  the R² score of 0.837 implies that the model accounts for nearly 84% of the variance in the target variable, demonstrating a strong explanatory power. 
  
![rf_result](https://github.com/user-attachments/assets/b5b4edb5-f782-43e1-b8a2-0c62f44b2aca)
Lookign at actual vs predicted the  values , we can closely track the actual market values. 

Taken together, these results suggest that the Random Forest model provides an improvement insights in the dataset. 

### XGBoost: 
An XGBoost Regressor model is built and evaluated. Feature importances are visualized as well.

### Deep Neural Network (DNN): 
Two versions of a Sequential Keras model are built with different architectures and training parameters (including Batch Normalization, Dropout, 
L2 regularization, and learning rate reduction). Both DNNs are trained and evaluated. Training history (loss and MAE), actual vs. predicted values, residuals, and error distributions are visualized for the DNN models.
