import numpy as np
import pandas as pd
import sklearn
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('archive/amp.csv', sep=',', error_bad_lines=False, low_memory=False, parse_dates=[18,29,31,32,37,41,42,44])
#dataset.shape
dataset.head()
#dataset.describe()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#data types
type(dataset)
dataset.dtypes

#count Nulls per column
percent_missing = dataset.isnull().sum() * 100 / len(dataset)
missing_value_df = pd.DataFrame({'column_name': dataset.columns, 'percent_missing': percent_missing})

#clean up missing data

#univariable analysis (Column Level)
Index_keep = missing_value_df.query('percent_missing < 75').index.tolist() 
column_keep = Index_keep
dataset = dataset[column_keep]
dataset.columns

#multivariable analysis (MCAR, MAR, NMAR) (Row Level)

#Pairwise deletion

#Listwise deletion


#Mean/Median Imputation


#Regression Imputation


#Random Forest Imputation


#Hot/Cold Desk Imputation


#Case Substitution

#Prior Knowledge

#Expectation Maximation


#Maximum Likelihood



#Imputations per missing data type