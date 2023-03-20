from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

iris = datasets.load_iris()

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL:', s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')
print(df.head(5))
print('vs')
print(iris["data"][0:5])

X = iris.data[:, [2, 3]]
y = iris["target"]

print(X[0], y[0])
print('Unique y outputs: ', len(np.unique(y)), ' with values ', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=RANDOM_STATE, stratify=y, shuffle=True)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train[0:5], ' vs ', X_train_std[0:5])

ppn = Perceptron(eta0=0.1, random_state=RANDOM_STATE)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Comparing pred vs real')

for index, pred in enumerate(y_pred):
    if pred == y_test[index]:
        print('\033[94m', pred, ' vs ', y_test[index], ' correct\033[0m')
    else:
        print('\033[91m', pred, ' vs ', y_test[index], ' error\033[0m')

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Another way to do the same things')
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

# graph the results
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
