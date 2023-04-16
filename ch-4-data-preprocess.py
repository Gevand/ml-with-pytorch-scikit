import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from io import StringIO
csv_data = \
    '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df.head())
print(df.isnull().sum())

# drop all rows that have atleast on missing data in a column
print(df.dropna(axis=0))
# drop all columns that have at least on missing data in a row
print(df.dropna(axis=1))
# drop all rows where everythign is null
print(df.dropna(how='all'))
# drop rows that have fewer than 4 real values
print(df.dropna(thresh=4))
# only worry about rows with C column empty
print(df.dropna(subset=['C']))

# for categorial data use "most_frequent" strategy
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)
