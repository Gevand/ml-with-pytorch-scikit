
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from mlp import NeuralNetMLP, int_to_onehot


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(X.values)
X = X.values
y = y.astype(int).values

print(X.shape, y.shape)
X = ((X / 255.) - .5) * 2  # normalize between -1 and 1
# fig, ax = plt.subplots(nrows=2, ncols=5,
#                        sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(10):
#     img = X[y == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(25):
#     img = X[y == 7][i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

model = NeuralNetMLP(num_features=28*28,
                     num_hidden=50,
                     num_classes=10)
num_epochs = 50
minibatch_size = 100


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]


for i in range(num_epochs):
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break

print(X_train_mini.shape, y_train_mini.shape)


def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(
        targets, num_labels=num_labels
    )
    return np.mean((onehot_targets - probas)**2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.1f}')