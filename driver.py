from itertools import count
from os import O_NOINHERIT
from typing import Counter
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
from pandas.core.frame import DataFrame
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import OrderedDict
from data_processing.input_manipulation import *
import os
import glob


# Load the data, and separate the target
data_path = 'marketing_campaign.csv'
marketingData = setUpData(data_path)

# Create Exclusion Lists to Change Data Cardinality
marriageExclusionList = ['Absurd', 'YOLO']
educationExclusionList = ['2n Cycle', 'Basic']

# Change Column Values to 'Other'
changeColumnValues(marketingData, educationExclusionList, 'Education')
changeColumnValues(marketingData, marriageExclusionList, 'Marital_Status')

# Output .csv File Into data_output_csv Folder
# marketingData.to_csv('data_output_csv/marketing_data_output.csv')

# # Create Y
y = marketingData.MntWines

# # Create X 
features = ['Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'NumWebVisitsMonth']

# Separate Features Into List Variables for Categorical and Numerical Columns
categorical_cols = ['Education', 'Marital_Status']
numerical_cols = ['Income', 'Kidhome', 'Teenhome', 'NumWebVisitsMonth']

# # Check Cardinality and Print
# for feature in features:
#     cardinalityCount(marketingData, feature)

# Results
# Column                Cardinality:             dtype:     Note: 
# Education             Cardinality is: 4        String     Needs to be Encoded
# Marital_Status        Cardinality is: 7        String     Needs to be Encoded
# Income                Cardinality is: 1975     int
# Kidhome               Cardinality is: 3        int
# Teenhome              Cardinality is: 3        int
# NumWebVisitsMonth     Cardinality is: 16       int 

# Select Columns Corresponding to Feature and Preview Data
X = marketingData[features]

# Check Code
# col_mask = X.isnull().any(axis=0)
# print(col_mask)

# Split Into Validation and Training Data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=0)



# ***************************************************************************************************************************
#                                                       Transformers                                                        #
# ***************************************************************************************************************************
# Numerical Transformer for Preprocessing Numerical Data
numerical_transformer = SimpleImputer(strategy='constant')

# Categorical Transformer for Preprocessing Categorical Data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# ***************************************************************************************************************************
#                                              Random Forest Model w/ Pipeline                                              #
# ***************************************************************************************************************************
# Define a Random Forest Model
rfmModel = RandomForestRegressor(n_estimators=100, random_state=0)

# Create Pipeline for RGM
rfm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', rfmModel)
])

# Fit Pipeline/Model
rfm_pipeline.fit(X_train, Y_train)

# Get Predictions
rfm_predictions = rfm_pipeline.predict(X_valid)



# Evaluate MAE for RGM 
rfm_score = -1 * cross_val_score(rfm_pipeline, X, y,
                                 cv=5,
                                 scoring='neg_mean_absolute_error')

print('Average RFM MAE:', rfm_score.mean())


# ***************************************************************************************************************************
#                                              XGB Regressor Model w/ Pipeline                                              #
# ***************************************************************************************************************************
# Define a XGB Regressor Model
xgbModel = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

# Create Pipeline for RGM
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', xgbModel)
])

# Fit Pipeline/Model
xgb_pipeline.fit(X_train, Y_train)

# Get Predictions
xgb_predictions = xgb_pipeline.predict(X_valid)

# Evaluate MAE for RGM 
# Evaluate MAE for RGM 
xgb_score = -1 * cross_val_score(xgb_pipeline, X, y,
                                 cv=5,
                                 scoring='neg_mean_absolute_error')

print('Average XGB MAE:', xgb_score.mean())






# ################################################ Debugging #####################################################
# # Copy Paste Whatever Needs to Be Checked in the dataCheck list
# # X.head(), X_train.head(), X_valid.head(), Y_train.head(), Y_valid.head(), OH_cols_train, OH_cols_valid, OH_X_train, num_X_train, num_X_valid, OH_X_valid

# # Array Storing All Datasets w/ Functions
# # dataCheck = [X.head(), X_train.head(), X_valid.head(), Y_train.head(), Y_valid.head(), OH_cols_train, OH_cols_valid, OH_X_train, num_X_train, num_X_valid, OH_X_valid]

# # Check All Datasets in dataCheck
# # for data in dataCheck:
# #     print("\n")
# #     print(data)

# # Check a Single Dataset
# # print(OH_cols_valid)

###################################################################################################################



