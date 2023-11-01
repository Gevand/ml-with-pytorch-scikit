import matplotlib.pyplot as plt
import numpy as np

# logistic function is f(z) = 1 / (1 + e ^ -z)
# softmax = f(z) = 1 / (sum(np.exp(z)))  -- sum is Sigma

X = np.array([1, 1.4, 2.5])
w = np.array([0.4, 0.3, 0.5])


def net_input(X, w):
    return np.dot(X, w)


def logistic(z):
    return 1.0 / (1 + np.exp(-z))


def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def tanh(z):
    # 2 × logistic(2𝑧) - 1
    # which becomx (e ^ z - e ^ (-z)) / (e ^ z + e ^ (-z))
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


print(f'P(y=1|x) = {logistic_activation(X, w):.3f}')

# W : array with shape = (n_output_units, n_hidden_units+1)
# note that the first column are the bias units
W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])
# A : data array with shape = (n_hidden_units + 1, n_samples)
# note that the first column of this array must be 1
A = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas = logistic(Z)
print('DOESNT SUM TO 1 -- Net Input: \n', y_probas)
y_probas = softmax(Z)
print('SOFTMAX SUMS TO 1 -- Net Input: \n', y_probas)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.plot(z, tanh_act,
         linewidth=3, linestyle='--',
         label='tanh')
plt.plot(z, log_act,
         linewidth=3,
         label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
