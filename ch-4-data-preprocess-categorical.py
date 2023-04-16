from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
# dealing with categorical data
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']

# ordinal category means it can be ordered, here M < L < XL
size_mapping = {'M': 1, 'L': 2, 'XL': 3}
df['size'] = df['size'].map(size_mapping)
print(df.head())

# inverse of size_mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))

# class mapping are not ordinal, they are nominal, so it doesn't matter which class gets which number
class_mapping = {label: idx for idx,
                 label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df.head())


# another way to do this with a label encoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))

# one hot encoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])])
print(c_transf.fit_transform(X).astype(float))

# another way to one hot encode built into pandas
print(pd.get_dummies(df[['price', 'color', 'size']]))

# sometimes its good to drop a the first column, as it makes the search space smaller
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

# same thing as above ^ but with one hot encoder from scikit learn
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])])
print(c_transf.fit_transform(X).astype(float))