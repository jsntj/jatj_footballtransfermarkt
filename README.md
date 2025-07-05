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
Linear Regression: As a foundational model, Linear Regression provided a baseline for performance. its performance was modest, achieving an R² score of approximately 0.46. 
The actual vs. predicted plot showed a tendency for the model to underpredict higher market values and overpredict lower ones, and the residuals exhibited a pattern, suggesting that the linear assumptions were not fully met.
![result linear](https://github.com/user-attachments/assets/0c9f3414-e5f7-4329-9431-707b7bdd9bbf)


### Random Forest: 
A Random Forest performed a significant improvement over Linear Regression. 
With an R² score of around 0.72, the Random Forest model explained larger portion of the variance in player market values. 
The feature importance analysis revealed key drivers, with contract_years_left, max_val, avg_val, and national_team_players appearing as highly influential features. 

![rf_result](https://github.com/user-attachments/assets/47296521-548f-447f-95b3-3281a96abc73)


The actual vs. predicted plot showed a better fit, particularly for values within the common range, although predicting extreme market values remains a problem.


### XGBoost: 
 further improved predictive performance, achieving the highest R² score among the traditional machine learning models at approximately 0.75. Similar to Random Forest, the feature importance reinforced the significance of contract duration and historical market value/transfer fees. The actual vs. predicted plot showed the best alignment with the actual values across the range, and the residuals were more tightly clustered around zero, indicating more accurate predictions.
![xgb_result](https://github.com/user-attachments/assets/444e52f5-976d-434f-bb88-6b6a483a04e4)


 
### Deep Neural Network (DNN): 
Two versions of a Sequential Keras model are built with different architectures and training parameters (including Batch Normalization, Dropout, 
L2 regularization, and learning rate reduction). Both DNNs are trained and evaluated. Training history (loss and MAE), actual vs. predicted values, residuals, and error distributions are visualized for the DNN models.
