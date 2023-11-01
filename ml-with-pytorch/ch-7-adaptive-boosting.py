# algorithm for adaptive boosting:
# draw a random subset of training examples, d1, without replacement from the training dataset, D, to train a weak learner, C1
# draw a second randomce training subset, d2, without replacement from the training dataset and add 50 percent of the example that were previously missclassified to train a weak learner C2.
# find the training examples, d3, in the training dataset, D, which C1 and C2 disagree upon to train a third weak learning, C3
# combine the weak leranings C1, C2, C3 via majority voting

# Ada boost does it a bit differently

# 1. create a weight vector W to uniform weights where they sum up to 1
# 2. for j in m boosting rounds, to the following
#   a. Train a weighted weak learner Cj = train(X, y, w)
#   b. predict y' = predict(Cj,X)
#   c. compute the error rate e = w * (y' != y)
#   d. compute teh boosting cooficient aj = .5 log (1 - e) / e
#   e. updtes the weights w = w x exp(-aj x y' x y).
#   f. normalize the weights to sum up to one.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# bellow is an example of 1 iteration of the boosting round, given some predicted vs actual values
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])

correct = y == yhat
w = np.full(y.shape, 1 / y.shape[0])
print(correct, w)

epsilon = w.dot(~correct)
alpha = .5 * np.log((1 - epsilon) / epsilon)
print(epsilon, alpha)

w = w * np.exp(-alpha * y * yhat)
w_normalized = w / np.sum(w)
print(w, w_normalized)

# using scikit learn to do the same thing
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash',
                   'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

# droping label 1
df_win = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']
            ].values  # two features only

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=1, stratify=y)

tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f'Decision tree train/test accuracies '
      f'{tree_train:.3f}/{tree_test:.3f}')

print('vs')

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(f'AdaBoost train/test accuracies '
      f'{ada_train:.3f}/{ada_test:.3f}')
