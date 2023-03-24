# logistic regression
from sklearn import datasets
from typing import List
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def logit(p):
    return math.log(p/(1-p))


def logistic_sigmoid_fn(z):
    return 1 / (1 + math.pow(math.e, -1 * z))


logit_x = [i for i in np.arange(.1, 1, .1)]
logit_y = []
sigmoid_x = [i for i in np.arange(-10, 10, .1)]
sigmoid_y = []
for i in logit_x:
    logit_y.append(logit(i))

#plt.scatter(logit_x, logit_y)
# plt.show()

for i in sigmoid_x:
    sigmoid_y.append(logistic_sigmoid_fn(i))

#plt.scatter(sigmoid_x, sigmoid_y)
# plt.show()


class LogisticRegressionGD_Geo:
    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = []
        self.b_ = 0

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n = len(y)
        self.w_ = np.random.randn(X.shape[1])
        self.b_ = np.float_(0)

        self.losses_ = []
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output))
                    - ((1 - y).dot(np.log(1 - output)))
                    / X.shape[0])
            self.losses_.append(loss)
        return self

    def loss_fn(self, target, output):
        return - target * np.log(output) - (1 - target) * np.log(1 - output)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(slef, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= .5, 1, 0)


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
iris = datasets.load_iris()

#binary classification, s only 2 different classes selected
X = iris.data[0:100, [2, 3]]
y = iris["target"][0:100]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=RANDOM_STATE, stratify=y, shuffle=True)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

log_reg_geo = LogisticRegressionGD_Geo(
    n_iter=1000, eta=0.5, random_state=5).fit(X_train_std, y_train)
y_pred = log_reg_geo.predict(X_test_std)
print('Comparing pred vs real')

for index, pred in enumerate(y_pred):
    if pred == y_test[index]:
        print('\033[94m', pred, ' vs ', y_test[index], ' correct\033[0m')
    else:
        print('\033[91m', pred, ' vs ', y_test[index], ' error\033[0m')
