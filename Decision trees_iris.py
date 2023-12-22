from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]   # we use only 2 columns to make the classification easier and to plot to look smaller
y = iris.target

print('class labels', np.unique(y))  # these classes are types of the iris flower

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)  # stratify y
print('label counts in y', np.bincount(y))                                               # to make the distribution even
print('label counts in y_train', np.bincount(y_train))
print('label counts in y_test', np.bincount(y_test))

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

tree = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=42)
tree.fit(X_train, y_train)
plot_tree(tree, filled=True, rounded=True, class_names=['Setosa', 'Versicolor', 'Virginica'],
          feature_names=['petal length', 'petal width'])

"""to plot the decision regions:"""
from mlxtend.plotting import plot_decision_regions
plt.figure()
plot_decision_regions(X_train, y_train, tree)
plt.xlabel('petal length in cm')
plt.ylabel('petal width in cm')
plt.legend(loc='best')
plt.show()