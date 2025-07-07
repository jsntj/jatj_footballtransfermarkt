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
fundamental as possible. Hence, we focused on these aspect while building :
- Try avoid any compromiesed dataset for modelling - for e.g. data leakeage due to features building or players' outliers
- Generate different models and Use same features - and observed their valuable features (except DNN). 
- Generate model that is as accurate as possible with as little as over/underfitting
- Get : different techniques to improved the model. RandomSearch.

## Data Loading and Preparation: 
To ensure everyone that approaches the notebook- we stored the dataset from kaggle in personal google drive folder and load them directly into the colab notebook. This process could be replaced with any loading process.
The script will work as long the dataset remains


### Data Splitting: 

The data is split into training and testing sets. 

## Feature Engineering:

A comprehensive set of features is engineered by merging the different dataframes. As one can see the datasets have Ids that could be merged/unioned. 

![kaggle](https://github.com/user-attachments/assets/f19c095f-e303-4ece-92f7-d45ab340bac9)

We decided to build features based on **The Players** and use market_value as the target. 

Player Demographics & Contract
- age: Player's age calculated from date of birth.
- contract_years_left: Years remaining on the player's contract.
- height_in_cm: Player's height in centimeters.
- is_foreigner: Whether the player’s citizenship differs from the club’s country.
- is_top_league: Whether the player’s club competes in a top-tier league (e.g., L1, GB1, ES1, IT1, FR1).

Player Performance (Aggregated)
- goals_per_90: Goals scored per 90 minutes played.
- assists_per_90: Assists per 90 minutes played.
- contrib_per_90: Combined goals and assists per 90 minutes.
- cards_per_90: Yellow and red cards per 90 minutes.
- appearances: Total number of games played.

Recent Form (Last 10 Appearances)
- recent_contrib: Total goals + assists in the last 10 matches.

Transfer History
- transfer_count: Number of transfers the player has had.
- max_fee: Highest transfer fee paid for the player.
- total_fees: Total sum of all transfer fees.
- avg_val: Average market value across transfer records.
- max_val: Maximum market value recorded.

Club Performance
- win_rate: Average win rate of the club.
- goals_scored: Average goals scored by the club per game.
- goals_conceded: Average goals conceded by the club per game.

Positional Flags
- is_forward: True if the player is a forward (CF, LW, RW, ST).
- is_midfielder: True if the player is a midfielder (CM, CAM, CDM, LM, RM).
- is_defender: True if the player is a defender (CB, LB, RB, WB).
- is_goalkeeper: True if the player is a goalkeeper (GK).

Club-Level Contextual Features
- squad_size: Number of players in the club's squad.
- average_age: Average age of players in the club.
- foreigners_percentage: Percentage of foreign players in the club.
- national_team_players: Number of players in the club who play for national teams.
- total_market_value_club: Total market value of the club.

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
![dnn v1 result2](https://github.com/user-attachments/assets/78c61bcc-1ecd-4162-935b-a2a4f9b9632e)


DNN V2, with additional layers, Batch Normalization, Dropout, and a learning rate scheduler, showed improved performance (R² ~ 0.67) compared to V1 and Linear Regression, but did not surpass the performance of the tree-based models (Random Forest and XGBoost) in this implementation. 
![dnnv2t_result](https://github.com/user-attachments/assets/e040810b-beba-44de-b912-62a8d2238d37)
![dnnv2result](https://github.com/user-attachments/assets/2744cf40-eced-4f6d-a2d7-64158647d467)


The training history plots for the DNNs provided insight into the learning process, showing how the loss and MAE decreased over epochs. The actual vs. predicted and residual plots for the DNNs indicated their ability to capture some non-linearities, with DNN V2 showing a better fit than V1. However, they are still under performed against xgb and rf. 

## DNN_v2 after hyperparameter tuning (optional)

To improve the performance of the initial DNN V2 model, we attempted hyperparameter tuning whioch was conducted using Keras Tuner's RandomSearch. 

The objective of the tuning process was to minimize the validation loss (val_loss) during training. The tuning process resulted in an improved DNN V2 model. Upon evaluation on the test set (X_test_processed), the tuned model achieved an R² score of 0.6979, a Mean Absolute Error (MAE) of 0.6671, and a Root Mean Squared Error (RMSE) of 0.8683.

### Overall result / conclusion
Random forest and XGb remains the top model for this dataset. After extensive Randomsearch the DNN_v2 still perform a disappointing below the XGB and RF. 

![model-result](https://github.com/user-attachments/assets/f1fd2d32-afba-417e-9973-2649799d1dcc)


## Hindsight
There could be improvement such as
- Hyperparameter tuning all the models
- Building more effective features after
- Inject additional players performance dataset from different sources
- Inject additional players or club popularity based on social media

