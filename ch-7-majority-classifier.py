from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.pipeline import _name_estimators
from sklearn.base import clone


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None) -> None:
        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }

    def fit(self, X, y):
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' "
                             f"or 'classlabel'"
                             f"; got (vote={self.vote})")
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and'
                             f' weights must be equal'
                             f'; got {len(self.weights)} weights,'
                             f' {len(self.classifiers)} classifiers')
        self.lablenc_ = LabelEncoder()  # labels need to start with 0
        self.lablenc_.fit(y)  # labels need to start with 0
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []  # fitted classifiers
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict_proba(self, X):
        # every classifier votes
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0,
                               weights=self.weights)  # apply the weights to each probability
        return avg_proba

    def predict(self, X):
        if self.vote == 'probability':
            # find the winner from average probabilityes
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # classlabel
            predictions = np.asarray([
                clf.predict(X) for clf in self.classifiers_
            ]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self.weights)
                ),
                axis=1, arr=predictions
            )
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(
                        deep=True).items():
                    out[f'{name}__{key}'] = value
            return out


iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test =\
    train_test_split(X, y,
                     test_size=0.5,
                     random_state=1,
                     stratify=y)


RANDOM_STATE = 1
clf1 = LogisticRegression(penalty='l2', C=0.001,
                          solver='lbfgs', random_state=RANDOM_STATE)
clf2 = DecisionTreeClassifier(
    max_depth=1, criterion='entropy', random_state=RANDOM_STATE)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train,
                             y=y_train, cv=10, scoring='roc_auc')
    print(
        f'ROC AUC: {scores.mean():.2f} 'f'(+/- {scores.std():.2f}) [{label}]')

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                            X=X_train,
                            y=y_train,
                            cv=10,
                            scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} '
        f'(+/- {scores.std():.2f}) [{label}]')
