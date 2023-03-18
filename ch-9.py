from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
import matplotlib.pyplot as plt
import pandas as pd


columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']

df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)

print(df.head(), df.shape)

df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})

print(df.isnull().sum())

# need to fix total bsmt st, it has 1 null value
# record 1341 has an issue
print(df.loc[df.isnull()['Total Bsmt SF'] == True].head())
df = df.dropna(axis=0)

scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()

slr = LinearRegression()
print(df[['Gr Liv Area']].values.shape, " vs ", df['Gr Liv Area'].values.shape,
      ' vs ', df[['Gr Liv Area']].shape, ' vs ',  df['Gr Liv Area'].shape)
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope ', slr.coef_[0])
print('Intercept ', slr.intercept_)

ransac = RANSACRegressor(LinearRegression(), max_trials=100,
                         min_samples=.95, residual_threshold=None, random_state=123)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

target = 'SalePrice'
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=123)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

ax1.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s',
            edgecolor='white',
            label='Test data')

ax2.scatter(
    y_train_pred, y_train_pred - y_train,
    c='steelblue', marker='o', edgecolor='white',
    label='Training data')
ax1.set_ylabel('Residuals')

for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)
plt.tight_layout()
plt.show()
