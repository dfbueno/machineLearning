# ***************************************************************************************************************************
#                                                         Imports                                                           #
# ***************************************************************************************************************************

from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank, duplicated
from scipy.sparse import data
from sklearn.utils import validation
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor


# ***************************************************************************************************************************
#                                                        Data Set-Up                                                        #
# ***************************************************************************************************************************

# Creates Dataframe
def setUpData(dataPath): 
    # Load the data, and separate the target
    dataFrame = pd.read_csv(dataPath)
    return dataFrame

# Creates Predicted Feature
def predictedFeature(dataFrame):
    # Create Predicted Feature
    predictedFeature = dataFrame.points 
    return predictedFeature

# Creates Predicting Features
def predictingFeatures(dataFrame, features):
    X = dataFrame[features]
    return X 


# ***************************************************************************************************************************
#                                                       Data Manipulation                                                   #
# ***************************************************************************************************************************

# Removes the Duplicates in a DataFrame
def duplicateRemoval (dataFrame): 
    dataFrame['is_duplicate'] = dataFrame.duplicated()
    dataFrame['is_duplicate'] = dataFrame.duplicated()
    dataFrame = dataFrame.drop_duplicates(subset='placeholder')
    dataFrame.reset_index(drop=True,inplace=True)
    dataFrame = dataFrame.drop('is_duplicate',axis=1)

def cardinalityReduction (dataFrame, column): 
    selectedColumn = dataFrame[f'{column}']
    cardinality = selectedColumn.unique()

    # print(cardinality) 

    # Get Counts For Each column
    cardinalitySeries = selectedColumn.value_counts(ascending=False) 

    # Grab Value Counts per Unique Entry -> Turn Into List -> Take Top 10 -> Replace Anything Not In Top 10
    cardinalityList = cardinalitySeries.index.tolist()

    # Create Exclusion List from Cardinality
    exclusionList = cardinalityList[10:]

    # Fill Data Frame With Other Using the Exclusion List
    dataFrame['column'] = dataFrame['column'].replace(exclusionList, 'Other')
    dataFrame['column'] = dataFrame['column'].fillna('Other')

    # Replacing the column Column w/ the New Country Data Adds Index Column "Unnamed: 0" -> Dropped
    dataFrame = dataFrame.drop('Unnamed: 0', axis='columns')

    return dataFrame


# ***************************************************************************************************************************
#                                                       Results Output                                                      #
# ***************************************************************************************************************************

# Takes Original DataFrame, Predictions, and Model Name -> Outputs a .csv With Predictions Concatenated Based On Predicted Feature (Points)
def outputCSV(originalDataFrame, predictions, modelName, validationData):
    validationData = validationData.copy()
    validationData['predictions'] = predictions
    validPredictions = pd.DataFrame(validationData['predictions'])
    modelPredictions = pd.merge(originalDataFrame, validPredictions, how = 'left', left_index = True, right_index = True)
    modelPredictions.to_csv(f"data_output_csv/{modelName}.csv")