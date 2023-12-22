import openpyxl as openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

training_data = pd.read_excel("./lab reg tree train.xlsx", engine='openpyxl',
                              usecols=lambda x: 'Unnamed' not in x)
test_data = pd.read_excel("./lab reg tree test.xlsx", engine='openpyxl',
                          usecols=lambda x: 'Unnamed' not in x)

training_data = training_data.dropna()  # clean up the missing values
test_data = test_data.dropna()
training_data = training_data._get_numeric_data()  # clean up all the text values
test_data = test_data._get_numeric_data()
# print(training_data.shape)
training_data.hist(figsize=(15, 15))
# plt.show()

target_variable_name = 'price'
training_values = training_data[target_variable_name]
training_points = training_data.drop(target_variable_name, axis=1)
# print(training_points.shape)

from sklearn import linear_model, ensemble
linear_regression_model = linear_model.LinearRegression()
random_forest_model = ensemble.RandomForestRegressor(random_state=42)

linear_regression_model.fit(training_points, training_values)
random_forest_model.fit(training_points, training_values)

test_values = test_data[target_variable_name]
test_points = test_data.drop(target_variable_name, axis=1)

# print(list(test_points)==list(training_points))  # double check if the X names correspond in test an train DS

test_predictions_linear = linear_regression_model.predict(test_points)
test_predictions_random_forest = random_forest_model.predict(test_points)

plt.figure(figsize=(7, 7))
plt.scatter(test_values, test_predictions_linear)  # our predictions
plt.plot([0, max(test_values)], [0, max(test_predictions_linear)], color='red')  # a line where true
                                                                                   # values and predictions are equal
plt.xlabel('actual price')
plt.ylabel('predicted price')
# plt.show()

plt.figure(figsize=(7, 7))
plt.scatter(test_values, test_predictions_random_forest)
plt.plot([0, max(test_values)], [0, max(test_predictions_random_forest)], color='red')
plt.xlabel('actual price', fontsize=20)
plt.ylabel('predicted price', fontsize=20)
# plt.show()

"""from visualization looks like the random forest did a better job, but need to check the metrics"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse_linear = mean_squared_error(test_values, test_predictions_linear)
mae_linear = mean_absolute_error(test_values, test_predictions_linear)
r2_linear = r2_score(test_values, test_predictions_linear)
mse_random_forest = mean_squared_error(test_values, test_predictions_random_forest)
mae_random_forest = mean_absolute_error(test_values, test_predictions_random_forest)
r2_random_forest = r2_score(test_values, test_predictions_random_forest)

print(f'mse {(mse_linear**0.5).__round__(2)},'
      f' mae {mae_linear.__round__(2)},'
      f' r2 {r2_linear.__round__(2)} - for linear')
print(f'mse {(mse_random_forest**0.5).__round__(2)},'
      f' mae {mae_random_forest.__round__(2)},'
      f' r2 {r2_random_forest.__round__(2)} - for random forest')