import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 10)

training_data = pd.read_csv("./cs-training.csv")
test_data = pd.read_csv("./cs-test.csv")
train_mean = training_data.mean()

from sklearn.model_selection import train_test_split
training_data, test_data = train_test_split(training_data, train_size=0.7)

"""to fill the missing values we can input the mean value in places"""
training_data.fillna(train_mean, inplace=True)
"""the target variable is the one where the default existence is described"""
target_variable_name = 'SeriousDlqin2yrs'
# print(training_data[target_variable_name].value_counts())
"""the 1 == True value is how many people have the default"""

training_values = training_data[target_variable_name]  # we input the target column to a new variable
# print(test_data[target_variable_name].nunique())
# drop the target from the og df
training_points = training_data.drop(target_variable_name, axis=1)

"""random forest method will be used"""

from sklearn import ensemble

# classifier is used for classification
random_forest_model = ensemble.RandomForestClassifier(n_estimators=100)  # n_estimators defs number of trees in forest
random_forest_model.fit(training_points, training_values)

"""the same mining is used for the testing data"""
test_data.fillna(train_mean, inplace=True)
test_values = test_data[target_variable_name]
test_points = test_data.drop(target_variable_name, axis=1)

"""predictions"""
test_predictions_random_forest = random_forest_model.predict(test_points)
# print(pd.value_counts(test_predictions_random_forest))  # how many defaults are in the predictions

from sklearn.metrics import confusion_matrix
random_forest_confusion_matrix = confusion_matrix(test_values, test_predictions_random_forest)
random_forest_confusion_matrix = pd.DataFrame(random_forest_confusion_matrix)

print(random_forest_confusion_matrix)
"""True positive (TP) -- an object of class 1 has been correctly labeled with label 1;
   False positive (False positive, FP) -- the object actually belongs to class 0, but is labeled 1;
   True negative (TN) -- the classifier correctly determined that the object of class 0 belongs to class 0;
   False negative (False negative, FN) -- the classifier labeled the object with label 0, but the object actually
   belongs to class 1."""

"""TN FP
   FN TP"""    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""MARGIN OF FALSE POSITIVE = FPR = FP/(FP+TN)
   MARGIN OF TRUE POSITIVE = TRP = TP/(TP+FN)
   
Note that FP+TN gives the total number of objects of class 0 , and TP+FN gives the total number of objects of class 1"""

"""The closer in general the ROC curve is to the upper left corner, the better the classification quality."""
from sklearn.metrics import roc_curve

test_probabilities = random_forest_model.predict_proba(test_points)
fpr, tpr, threshold = roc_curve(test_values, test_probabilities[:,1])

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import roc_auc_score

# to evaluate the model. 0.90 - 1.00 excellent;
# 0.80 - 0.90 good;
# 0.70 - 0.80 satisfactory;
# 0.60 - 0.70 poor;
# 0.50 - 0.60 very bad;
# 0.00 - 0.50 the classifier mixed up the labels

roc_auc_value = roc_auc_score(test_values, test_probabilities[:,1])
print("ROC-AUC for the test selection:", roc_auc_value.round(2))