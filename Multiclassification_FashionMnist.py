import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dictionary = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
    4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag',
    9: 'Ankle boot'
}

train = pd.read_csv("./fashion-mnist_train.csv")
test = pd.read_csv("./fashion-mnist_test.csv")
X_train = train.drop(columns='label')
y_train = train['label']
X_test = test.drop(columns='label')
y_test = test['label']

"""to visualize images is needed to convert the values of 1 image to a list"""

X_train_list = X_train.values.tolist()
X_test_list = X_test.values.tolist()

import numpy as np
import matplotlib.pyplot as plt

i = 56
plt.imshow(np.reshape(X_test_list[i], (28, 28, 1)))  # reshape the [image] to a 2d image (width, length, depth=1)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train, y_train)
score = lr.score(X_train, y_train)       # how well the model predicts the data
score_test = lr.score(X_test, y_test)

y_pred = lr.predict([X_test.iloc[i]])   # predict a value based on the model
y_pred = int(y_pred)

print('Item category prediction: ', dictionary[y_pred])

y_test = int(y_test.iloc[i])    # check the actual value
print('Item category from the dataset:', dictionary[y_test])

plt.show()