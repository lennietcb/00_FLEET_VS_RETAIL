import numpy as np
import pandas as pd
import sklearn
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('archive/uvs_report.csv', sep=',', error_bad_lines=False, low_memory=False)
dataset.shape
dataset.head()
dataset.describe()

#clean up missing data

#univariable analysis

#multivariable analysis (MCAR, MAR, NMAR)

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