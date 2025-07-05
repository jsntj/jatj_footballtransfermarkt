# jatj_football_transfermarkt

This repo is in essence, another version of JATJ's team approach to the model building our 2025 datascience course.  It summarize our Exploratory analysis of the dataset and how we decide to approach the modelling.  dtance.

dataset:
https://www.kaggle.com/datasets/davidcariboo/player-scores

# EDA

## Basic analysis - headcount
we start with understanding the dataset of players. as the result 

![download](https://github.com/user-attachments/assets/b433467f-a863-4c87-8e7c-2bead0edcfdd)

## Market value of players
### Age 
we look a bit of the age and we notice the market value is on general tilted up towards younger players group ( the 20s) with a number of players that goes to the extreme. This is assumed as star players/ outliers.  

![age market value](https://github.com/user-attachments/assets/177da79e-e53f-4b86-af90-7a488e7ffe0e)

![marketvalue_age](https://github.com/user-attachments/assets/0f89f911-0c04-44c5-a09a-f42b14a8269b)

### Position and overtime

Overtime the market value also increased - due to increased in popularity + inlfation. 

![marketvalue_position](https://github.com/user-attachments/assets/1543d376-478d-4472-baea-882e80b10792)

### Goals
Goals scored seems to be also indicator of market value but for attacked

![goal_market value](https://github.com/user-attachments/assets/a5d1b08d-92e5-446b-87c1-d8a2ea7bd9fc)


# Approaches
Based on the Eda and our level of knowledge of datascience and knowledge in football we decided to go as
fundamental as possible. Hence, we focused on these 3 aspect while building :
- Try avoid any compromiesed dataset for modelling - for e.g. data leakeage due to features building or players' outliers
- Generate different models and Use same features - and observed their valuable features (except DNN). 
- Generate model that is as accurate as possible with as little as over/underfitting
- Get : different techniques to improved the model. RandomSearch.

## Data Loading and Preparation: 
To ensure everyone that approaches the notebook- we stored the dataset from kaggle in personal google drive folder and load them directly into the colab notebook. This process could be replaced with any
### Data Splitting: 

The data is split into training and testing sets. 

## Feature Engineering:

A comprehensive set of features is engineered by merging the different dataframes. As one can see the datasets have Ids that could be merged/unioned. 

![kaggle](https://github.com/user-attachments/assets/f19c095f-e303-4ece-92f7-d45ab340bac9)

We decided to build features based on **The Players** and use market_value as the target. 

Deciding features were taken during our EDA. We understand **basic features** like age, height, position, foot are just low-hanging fruit so we added them. 
Performance metrics (goals, assists, cards per 90 minutes, appearances), generally should be tilted towards attacker but we keep goal in hope that we could improved it.
club performance (win rate, goals scored/conceded), 
transfer history (transfer count, fees), 
Other features like contract details, top leagues and foreign players. 
we also considered log-transformed to handle outlier for the market_value to handle outlier. 


## Preprocessing Pipeline: 

A preprocessing pipeline is defined using ColumnTransformer to handle categorical features (one-hot encoding) and numerical features (standard scaling). it was built standard. 


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
Two versions of a Sequential Keras model were explored. DNN V1, a simpler architecture, yielded performance comparable to Linear Regression (R² ~ 0.46). 
![dnn v1 result](https://github.com/user-attachments/assets/d51b8a97-2d10-4bf0-a070-949b0c91b52b)


DNN V2, with additional layers, Batch Normalization, Dropout, and a learning rate scheduler, showed improved performance (R² ~ 0.67) compared to V1 and Linear Regression, but did not surpass the performance of the tree-based models (Random Forest and XGBoost) in this implementation. 

![dnnv2result](https://github.com/user-attachments/assets/2744cf40-eced-4f6d-a2d7-64158647d467)


The training history plots for the DNNs provided insight into the learning process, showing how the loss and MAE decreased over epochs. The actual vs. predicted and residual plots for the DNNs indicated their ability to capture some non-linearities, with DNN V2 showing a better fit than V1. However, they are still under performed against xgb and rf. 
