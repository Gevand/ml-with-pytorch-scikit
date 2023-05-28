from scipy.special import comb
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.pipeline import _name_estimators
from sklearn.base import clone


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) *
             error**k *
             (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)


print(ensemble_error(n_classifier=11, error=0.25))
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range]

plt.plot(error_range, ens_errors,
         label='Ensemble error',
         linewidth=2)
plt.plot(error_range, error_range, linestyle='--',
         label='Base error', linewidth=2)

plt.grid(alpha=0.5)
plt.show()

# weigted voting

print('Pick 0 or 1 from three classifiers, where the third on has a weight of .6, the other two are .2. The .2 guys predicted 0, but .6 predicted 1, so 1 is still taken . mode([0, 0, 1, 1, 1]) -> ', np.argmax(
    np.bincount([0, 0, 1], weights=[.2, .2, .6])))

# assume we have three classifiers C1, C2, C3 for a binary classification problem  with probabilities
# 𝐶1(𝒙) → [0.9, 0.1], 𝐶2(𝒙) → [0.8, 0.2], 𝐶3(𝒙) → [0.4, 0.6]
# we weight the classifiers [.2, .2, .6], so whats the probability ensemble classifies as 0 or 1?
print(.2 * .9 + .8 * .2 + .4 * .6, ' = probability ensemble picks zero, p(0)')
print(.2 * .1 + .2 * .2 + .6 * .6, ' = probability ensemble picks one, p(1)')

# implement majority vote
ex = np.array([[.9, .1], [.8, .2], [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[.2, .2, .6])
print('Same thign as the calculation above -> ', p)



