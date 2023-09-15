import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchdata.datapipes.iter import FileOpener, IterableWrapper
import pandas as pd
import re
from collections import Counter, OrderedDict

from tqdm import tqdm
torch.manual_seed(1)
device = 'cuda'
torch.set_default_device(device)
basepath = '/mnt/d/Books/Python/ml-with-pytorch-scikit/aclImdb'
labels = {'pos': 1, 'neg': 0}
df = pd.read_csv(basepath + '/movie_data.csv', encoding='utf-8')
train_dataset, valid_dataset = random_split(
    df[0:25000].values.tolist(), [20000, 5000], generator=torch.Generator(device=device))
test_dataset = df[25000:].values.tolist()
print(len(train_dataset), ' vs ', len(valid_dataset))


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized


token_counts = Counter()
for line, label in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)
print('Vocab-size:', len(token_counts))

sorted_by_preq_tuples = sorted(
    token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_preq_tuples)
vocab = {}
vocab["<pad>"] = 0
vocab["<unk>"] = 1
for index, token in enumerate(ordered_dict):
    vocab[token] = index

print([vocab[token] for token in ['this', 'is', 'an', 'example']])


def text_pipeline(x): return [
    vocab[token] if token in vocab else vocab["<unk>"] for token in tokenizer(x)]


def label_pipeline(x): return 1. if x == 'pos' or x == 1 else 0.


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _text, _label in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list, lengths


dataloader = DataLoader(train_dataset, batch_size=4,
                        shuffle=False, collate_fn=collate_batch)

text_batch, label_batch, length_batch = next(iter(dataloader))
print(text_batch)
print(label_batch)
print(length_batch)
print(text_batch.shape)

batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size,
                      shuffle=True, generator=torch.Generator(device=device), collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, generator=torch.Generator(device=device), batch_size=batch_size,
                      shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, generator=torch.Generator(device=device), batch_size=batch_size,
                     shuffle=False, collate_fn=collate_batch)


embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
text_encoded_input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 0]])
print(embedding(text_encoded_input.to(device)))


class RNN_EXAMPLE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        # we use the final hidden state from the last hidden layer as the input to the fully connected layer
        out = hidden[-1, :, :]
        out = self.fc(out)
        return out


model = RNN_EXAMPLE(64, 32)
print(model)
result = model(torch.randn(5, 3, 64))
print(result)


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths: torch.Tensor):
        out = self.embedding(text)
        lengths_cpu = lengths.to('cpu')
        torch.set_default_device('cpu')
        out = nn.utils.rnn.pack_padded_sequence(
            input=out, lengths=lengths_cpu.numpy(), enforce_sorted=False, batch_first=True)
        torch.set_default_device(device)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim,
            rnn_hidden_size, fc_hidden_size)
model.to(device)
test = 12000 / len(dataloader.dataset)


def train(model: RNN, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.BCELoss):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in tqdm(dataloader, unit="batch", total=len(dataloader)):
        optimizer.zero_grad()
        pred: torch.Tensor = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() ==
                      label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    acc_train, loss_train = train(
        model=model, dataloader=train_dl, optimizer=optimizer, loss_fn=loss_fn)
    print(f'Epoch {epoch} accuracy: {acc_train:.4f} val_accuracy: {0:.4f}')

print('----------------------- Trying Bi-Directionaly RNN ----------------------------')
class BiDirectional_RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size,
                           batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size * 2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths: torch.Tensor):
        out = self.embedding(text)
        lengths_cpu = lengths.to('cpu')
        torch.set_default_device('cpu')
        out = nn.utils.rnn.pack_padded_sequence(
            input=out, lengths=lengths_cpu.numpy(), enforce_sorted=False, batch_first=True)
        torch.set_default_device(device)
        out, (hidden, cell) = self.rnn(out)
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)
model = BiDirectional_RNN(vocab_size, embed_dim,
            rnn_hidden_size, fc_hidden_size)
model.to(device)
test = 12000 / len(dataloader.dataset)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    acc_train, loss_train = train(
        model=model, dataloader=train_dl, optimizer=optimizer, loss_fn=loss_fn)
    print(f'Epoch {epoch} accuracy: {acc_train:.4f} val_accuracy: {0:.4f}')