# jatj_football_transfermarkt

This repo is in essence, another version of JATJ's team approach to the model building our 2025 datascience course.  It summarize our Exploratory analysis of the dataset and how we decide to approach the modelling.  dtance.

dataset:
https://www.kaggle.com/datasets/davidcariboo/player-scores

# EDA

## Basic analysis - headcount
we start with understanding the dataset of players. as the result 

![download](https://github.com/user-attachments/assets/b433467f-a863-4c87-8e7c-2bead0edcfdd)

## Market value
we look a bit of the age and we notice the market value is on general tilted up towards younger players with a number of players that goes to the extreme. This is assumed as star players 

![age market value](https://github.com/user-attachments/assets/177da79e-e53f-4b86-af90-7a488e7ffe0e)

Goals scored seems to be also indicator of market value but for attacked

![goal_market value](https://github.com/user-attachments/assets/a5d1b08d-92e5-446b-87c1-d8a2ea7bd9fc)


# Approaches
Based on the Eda and our level of knowledge of datascience and knowledge in football we decided to go as
fundamental as possible. Hence, we foucs on these 3 aspect while building :
- Try to remove star players as our outlier by not having players id.
- generate different models and Use same features - and observed their valuable features (except DNN)
- Generate model that is as accurate as possible with as little as  over/underfitting

## Data Loading and Preparation: 
To ensure everyone that approaches the notebook- we stored the dataset from kaggle in personal google drive folder and load them directly into the colab notebook. This process could be replaced with any

## Feature Engineering:

A comprehensive set of features is engineered by merging the different dataframes. These features include player attributes (age, height, position, foot), performance metrics (goals, assists, cards per 90 minutes, appearances), club performance (win rate, goals scored/conceded), transfer history (transfer count, fees), contract details, and flags for top leagues and foreign players. The target variable, market value, is also log-transformed to handle outliers.

## Data Splitting: 

The data is split into training and testing sets. 

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
