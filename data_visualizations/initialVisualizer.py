# ***************************************************************************************************************************
#                                                   Imports                                                                 #
# ***************************************************************************************************************************
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.matrix import heatmap
import itertools
import os
import glob

# Data Path Import
data_path = 'data_output_csv/marketing_data_output.csv'

# Set All Indexes to All Features to be Explored
education_data = pd.read_csv(data_path, index_col='Education')
marital_data = pd.read_csv(data_path, index_col='Marital_Status')
income_data = pd.read_csv(data_path, index_col='Income')
kidhome_data = pd.read_csv(data_path, index_col='Kidhome')
teenhome_data = pd.read_csv(data_path, index_col='Teenhome')
recency_data = pd.read_csv(data_path, index_col='Recency')
numwebvisitsmonth_data = pd.read_csv(data_path, index_col='NumWebPurchases')


# ***************************************************************************************************************************
#                                      Bar Plot for Average Amount of Wine Purchased by Education                           #
# ***************************************************************************************************************************
plt.figure(figsize=(10,6))
plt.ylim(0,500)
plt.title('Wine Purchased by Education Bar Plot')
sns.barplot(x=education_data.index, y=education_data['MntWines'])
plt.ylabel('Amount of Wine')
plt.xlabel('Education')
plt.show()

# ***************************************************************************************************************************
#                                      Bar Plot for Average Amount of Wine Purchased by Relationship Status                                  #
# ***************************************************************************************************************************
plt.figure(figsize=(10,6))
plt.ylim(0,500)
plt.title('Wine Purchased by Marriage Status Bar Plot')
sns.barplot(x=marital_data.index, y=marital_data['MntWines'])
plt.ylabel('Amount of Wine')
plt.xlabel('Relationship Status')
plt.show()

# ***************************************************************************************************************************
#                                      Scatter Plot for Amount of Wine Purchased by Income                                  #
# ***************************************************************************************************************************
plt.title('Amount of Wine Purchased by Income')
sns.scatterplot(x=income_data.index, y=income_data['MntWines'])
plt.show()

# ***************************************************************************************************************************
#                                    Bar Plot for Amount of Wine Purchased by Number of Children                            #
# ***************************************************************************************************************************

plt.figure(figsize=(10,6))
plt.ylim(0,500)
plt.title('Wine Purchased by Number of Children')
sns.barplot(x=kidhome_data.index, y=kidhome_data['MntWines'])
plt.ylabel('Amount of Wine')
plt.xlabel('Number of Children')
plt.show()

# ***************************************************************************************************************************
#                                    Bar Plot for Amount of Wine Purchased by Number of Teens                               #
# ***************************************************************************************************************************
plt.figure(figsize=(10,6))
plt.ylim(0,500)
plt.title('Wine Purchased by Number of Teens')
sns.barplot(x=kidhome_data.index, y=kidhome_data['MntWines'])
plt.ylabel('Amount of Wine')
plt.xlabel('Number of Teens')
plt.show()

# ***************************************************************************************************************************
#                                      Scatter Plot for Amount of Wine Purchased by Recency                                 #
# ***************************************************************************************************************************
plt.title('Amount of Wine Purchased by Recency')
sns.scatterplot(x=recency_data.index, y=recency_data['MntWines'])
plt.show()


# ***************************************************************************************************************************
#                         Scatter Plot for Amount of Wine Purchased by Number of Web Visits Per Month                       #
# ***************************************************************************************************************************
# plt.title('Amount of Wine Purchased by Recency')
# sns.scatterplot(x=numwebvisitsmonth_data.index, y=numwebvisitsmonth_data['MntWines'])
# plt.show()

plt.figure(figsize=(10, 6))
plt.ylim(0, 800)
plt.title('Wine Purchased by Number of Number of Web Visits Per Month')
sns.barplot(x=numwebvisitsmonth_data.index, y=numwebvisitsmonth_data['MntWines'])
plt.ylabel('Amount of Wine')
plt.xlabel('Number of Web Visits per Month')
plt.show()