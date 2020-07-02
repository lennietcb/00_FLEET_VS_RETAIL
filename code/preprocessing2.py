import numpy as np
import pandas as pd
import pickle
from scipy.optimize import fsolve
import sklearn
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('archive/uvs_report.csv', sep=',', error_bad_lines=False, low_memory=False)
dataset.shape
dataset.head()

#Trim target values
#dataset.sales_grp = dataset.sales_grp.str.strip()

#Replace string with binary
#dataset = dataset.replace({'sales_grp': {'RETAIL':0,'FLEET':1}})

#Rename target Column
#dataset = dataset.rename(columns={"sales_grp": "y"})

#Visualizations
pt= sns.countplot(x='y', data=dataset, palette='hls')
plt.show()
plt.savefig('count_plot')

fig2 = pd.crosstab(dataset.vehicle_grp,dataset.y).plot(kind='bar')
plt.title('Sales for Vehicle Group')
plt.xlabel('Vehicle')
plt.ylabel('Frequency of Vehicle Group')
plt.savefig('Frequency_vehicle_group')

table=pd.crosstab(dataset.cond_code,dataset.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Sales by Condition Code')
plt.xlabel('Sales')
plt.ylabel('Condition Codes')
plt.savefig('sales_by_condition_code')

#Describe
dataset.describe()

#Frecuency
dataset.groupby('y').mean()

#Analyze data types
dataset.dtypes

#Cast some Data Type (int, float)
dataset['Ask'] = dataset.Ask.astype(int)
dataset['Take'] = dataset.Take.astype(int)
dataset['Proceed_Over_Ask'] = dataset.Proceed_Over_Ask.astype(float)
dataset['Proceed_Over_Take'] = dataset.Proceed_Over_Take.astype(float)

#Nulls by Columns
percent_missing = dataset.isnull().sum() * 100 / len(dataset)
missing_value_df = pd.DataFrame({'column_name': dataset.columns, 'percent_missing': percent_missing})
missing_value_df

#Remove features before Model
#dataset = dataset.drop(['vehicle_num', 'MEQ_NUMBER__c', 'VehicleId','sale_date','accpt_date','Pricedate','CUSTOMER_NUMBER__c','CUSTOMER_NAME__c'], axis = 1)
dataset.head()

#Shape
dataset.shape

#IQR
def iqr_metric (df):
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1
  df_num_iqr = df[~((df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR))).any(axis=1)]
  return df_num_iqr 

dataset = iqr_metric (dataset)
dataset.shape

#Frecuency Table
def freq_table (df,colname):
      return df[colname].value_counts(normalize =True).reset_index(name='count')

#Replace dataset name and catcol names
catcol = ['vehicle_grp','age_sale_grp','cond_code','LOCATION_CODE__c']
df_cat= dataset[catcol]

for colname in df_cat:
  print(colname)
  print(freq_table(dataset, colname))

#Frecuency of cond_code
frequency = dataset['cond_code'].value_counts()
frequency

#Min and Max
minmax = dataset.agg(['min','max'])
minmax

#Replacing noise values
dataset = dataset[dataset["cond_code"] > 0]
dataset = dataset[dataset["proceeds"] >= 100]

#Dummies
cat_vars=['vehicle_grp','age_sale_grp']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(dataset[var], prefix=var)
    data1=dataset.join(cat_list)
    dataset=data1
cat_vars=['vehicle_grp','age_sale_grp']
data_vars=dataset.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=dataset[to_keep]
data_final.columns.values

#Target Distribution
frequency = dataset['y'].value_counts(normalize =True).reset_index(name='percent') 
frequency * 100

#Over-sampling using SMOTE
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

print("length of oversampled data is ",len(os_data_X))
print("Number of RETAIL in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of FLEET",len(os_data_y[os_data_y['y']==1]))
print("Proportion of RETAIL data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of FLEET data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]

freq = os_data_y.y.value_counts()
print(freq)

#Shape
dataset.shape

#Save to pickle file
import pickle

dataset.to_pickle('archive/prepared_data.pickle')
