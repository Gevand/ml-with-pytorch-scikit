import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pandas
import numpy as np
from torch.distributions.categorical import Categorical

device = 'cuda'
torch.set_default_device(device)

with open('ch-15/1268-0.txt', 'r', encoding="utf8") as fp:
    text = fp.read()

start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')
text = text[start_indx:end_indx]
char_set = set(text)

print('Total Length:', len(text))
print('Unique Characters:', len(char_set))

chars_sorted = sorted(char_set)
char2int = {ch: i for i, ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)

print('Text encoded shape:', text_encoded.shape)
print(text[:15], '== Encoding ==>', text_encoded[:15])
print(text_encoded[15:21], '== Reverse ==>',
      ''.join(char_array[text_encoded[15:21]]))

for ex in text_encoded[100:150]:
    print('{} -> {}'.format(ex, char_array[ex]))

seq_length = 40
prediction_size = 1
chunk_size = seq_length + prediction_size
text_chunks = [text_encoded[i:i + chunk_size]
               for i in range(len(text_encoded) - chunk_size)]


class TextDataset(Dataset):
    def __init__(self, text_chunks) -> None:
        super().__init__()
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()


seq_dataset = TextDataset(torch.tensor(text_chunks))


for i, (seq, target) in enumerate(seq_dataset):
    print(' Input (x): ', repr(''.join(char_array[seq.cpu()])))
    print(' Target (y): ', repr(''.join(char_array[target.cpu()])))
    print('-------------')
    if i == 1:
        break

batch_size = 64
torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size,
                    shuffle=True, drop_last=True, generator=torch.Generator(device=device))


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell


vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000

path = 'ch-15/character-gen.ph'
if os.path.isfile(path):
    model = torch.load(path)
else:
    for epoch in range(num_epochs):
        hidden, cell = model.init_hidden(batch_size)
        seq_batch, target_batch = next(iter(seq_dl))
        optimizer.zero_grad()
        loss = 0
        for c in range(seq_length):
            pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
            loss += loss_fn(pred, target_batch[:, c])
        loss.backward()
        optimizer.step()
        loss = loss.item() / seq_length
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {loss:.4f}')
    torch.save(model, path)


def sample(model, starting_str, len_generated_text=500, scale_factor=1.0):
    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str
    model.eval()
    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str)-1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])
    return generated_str


def sample_top_p(model, starting_str, len_generated_text=500, top_n=10):
    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str
    model.eval()
    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str)-1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        logits_sort, logits_idx = torch.sort(logits, dim=-1, descending=True)
        logits_sort=logits_sort[0:top_n]
        logits_idx=logits_idx[0:top_n]
        m = Categorical(logits=logits_sort)
        last_char = logits_idx[m.sample()]
        generated_str += str(char_array[last_char])
    return generated_str


print(sample(model, starting_str='The island'))
print('-----------+++++++++++++++--------------')
print(sample_top_p(model, starting_str='The island'))