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
from collections import OrderedDict

from data_processing.input_manipulation import *

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

# # Check Cardinality and Print
# for feature in features:
#     cardinalityCount(marketingData, feature)

# Results
# Column              Cardinality:             dtype:     Note: 
# Education           Cardinality is: 4        String     Needs to be Encoded
# Marital_Status      Cardinality is: 7        String     Needs to be Encoded
# Income              Cardinality is: 1975     int
# Kidhome             Cardinality is: 3        int
# Teenhome            Cardinality is: 3        int
# NumWebVisitsMonth   Cardinality is: 16       int 


# Select Columns Corresponding to Feature and Preview Data
X = marketingData[features]

# Check Code
# col_mask = X.isnull().any(axis=0)
# print(col_mask)

# Split Into Validation and Training Data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=0)

# Gets List of Categorical Columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Low Cardinality Columns (Will Be One-Hot Encoded)
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# High Cardinality Columns (To Be Dropped)
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

# Apply One-Hot Encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding Removed Index...Put It Back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove Categorical Columns (Replace w/ One-Hot Encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add One_Hot Encoded Columns to Numerical Features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Imputation After One-Encoding Process 
my_imputer = SimpleImputer()
imputed_OH_X_train = pd.DataFrame(my_imputer.fit_transform(OH_X_train))
imputed_OH_X_valid = pd.DataFrame(my_imputer.transform(OH_X_valid))

# Fill in Lines Below -> Imputation Removed Columns -> Put Them Back
imputed_OH_X_train.columns = OH_X_train.columns 
imputed_OH_X_valid.columns = OH_X_valid.columns

# Define a Random Forest Model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(imputed_OH_X_train, Y_train)
rf_val_predictions = rf_model.predict(imputed_OH_X_valid)
rf_val_mae = mean_absolute_error(rf_val_predictions, Y_valid)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



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

# #################################################################################################################

# # The current problem as it stands is that during the One-Hot Encoding process that the data is for some reason
# # converted to a datatype that extends past the size of float32/infinity or to NaN. The other thing that is confusing
# # is the error that references a piece of data..."Graduation" inside of the Education column and technically that should've been
# # One-Hot Encoded and should be within 4 bytes. 

# # datascience.stackexchange.com/questions/11928/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtypefloat32

# # I referenced the above link, but my problem is not checking the data, it's one-hot encoding properly. 

# # I included a little debugging section to look at all of the data at once and see which of the validation and training sets are 
# # causing the problem... 



