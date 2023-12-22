import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

pd.set_option('display.max_columns', 10)  # print 10 columns
housing = fetch_california_housing()
print(type(housing))
print(housing.keys())  # what are the keys of a dataframe
print(type(housing.data), type(housing.target))  # data types
print(housing.DESCR)  # info about dataset
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
print(housing_df)
housing_df['Population'] = housing.target  # add a new target variable
print(housing_df.describe().round(2))  # describe a dataset
print(housing_df.isnull().sum())  # how many missed values in a dataset. here are 0 missing values

# EXPLORATORY DATA ANALYSIS

corr_matrix = housing_df.corr().round(2)
print(corr_matrix)
x1 = housing_df['MedInc']
x2 = housing_df['AveBedrms']
y = housing_df['Population']
plt.figure(figsize=(8, 5))
plt.scatter(x1, y)
plt.xlabel('Median Income')
plt.ylabel('Population')

# X SHOULD BE A DATAFRAME, Y A SERIES
X = housing_df[['MedInc']]  # [['...']] create a df, can be [['ab', 'bc', 'cd']]
X = np.array(X)
y = housing_df['Population']  # [] create a series
y = np.array(y)
print(type(X), type(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# X_train = X.reshape(-1, 1)
# y_train = y[0::2]
#
# X_test = X.reshape(-1, 1)
# y_test = y[1::2]
#
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_test_pred = lr.predict(X_test)  # PREDICTIONS ARE USUALLY MADE WITH TEST DATA!!!!
print(type(y_test_pred), 'y_train_pred')  # have to be np arrays
print(type(X_train), 'X_train')
print(X_test.shape)  # dimensions of an array
print(y_test_pred.shape)
plt.plot(X_test, y_test_pred, color='black')

from sklearn.metrics import mean_squared_error

print('RMSE=', np.sqrt(mean_squared_error(y_test, y_test_pred)).round(2))

from sklearn.metrics import r2_score

print('R2=', r2_score(y_test, y_test_pred).round(2))  # how well observed outcomes are replicated by the model
# proportion of total variation of outcomes explained by the model

plt.show()