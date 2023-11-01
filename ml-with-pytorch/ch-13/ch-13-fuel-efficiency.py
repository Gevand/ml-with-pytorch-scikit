from torch.nn.functional import one_hot
import sklearn.model_selection
import sklearn
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

column_names = ['MPG', 'Cylinders', 'Displacement',
                'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv('ch-13/auto-mpg.data', names=column_names,
                 na_values='?', comment='\t', sep=' ', skipinitialspace=True)
df = df.dropna()
df = df.reset_index(drop=True)
print(df.head())


df_train, df_test = sklearn.model_selection.train_test_split(
    df, train_size=.8, random_state=1)
train_stats = df_train.describe().transpose()
numeric_column_names = ['Cylinders', 'Displacement',
                        'Horsepower', 'Weight', 'Acceleration']
df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (
        df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (
        df_test_norm.loc[:, col_name] - mean) / std
print(df_train_norm.tail())

# model year bucketed before 73, between 73-76, between 76 and 79, after 79
boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm['Model Year'].values)
df_train_norm['Model Year Bucketed'] = torch.bucketize(
    v, boundaries, right=True)
v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize(
    v, boundaries, right=True)
numeric_column_names.append('Model Year Bucketed')
print(df_train_norm.tail())

# one hot encode the Origin
# first get the number of unique values
total_origin = len(set(df_train_norm['Origin']))
origin_encoded = one_hot(torch.from_numpy(
    df_train_norm['Origin'].values % total_origin))  # if you don't modul0, one hot will be four dimensional [0,1,2,3] even though we only have category 1,2,3. So you have to turn 1 into 0, 2 into 1, and 3 into 2. Could have just subracted 1 instead of this modulo stuff
print(origin_encoded)
x_train_numeric = torch.tensor(
    df_train_norm[numeric_column_names].values)
x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()
origin_encoded = one_hot(torch.from_numpy(
    df_test_norm['Origin'].values) % total_origin)
x_test_numeric = torch.tensor(
    df_test_norm[numeric_column_names].values)
x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()

train_ds = TensorDataset(x_train, y_train)
batch_size = 8
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

hidden_units = [8, 4]
input_size = x_train.shape[1]
all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit
all_layers.append(nn.Linear(hidden_units[-1], 1))
model = nn.Sequential(*all_layers)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
torch.manual_seed(1)

num_epochs = 200
log_epochs = 20

for epoch in range(num_epochs):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss '
              f'{loss_hist_train/len(train_dl):.4f}')

with torch.no_grad():
    pred = model(x_test.float())[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')
