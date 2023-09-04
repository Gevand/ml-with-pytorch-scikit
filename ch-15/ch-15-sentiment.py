import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchdata.datapipes.iter import FileOpener, IterableWrapper
import pandas as pd
import re
from collections import Counter, OrderedDict
torch.manual_seed(1)

basepath = '/mnt/d/Books/Python/ml-with-pytorch-scikit/aclImdb'
labels = {'pos': 1, 'neg': 0}
df = pd.read_csv(basepath + '/movie_data.csv', encoding='utf-8')
train_dataset, valid_dataset = random_split(
    df[0:25000].values.tolist(), [20000, 5000])
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
print('Vocam-size:', len(token_counts))

sorted_by_preq_tuples = sorted(
    token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_preq_tuples)
vocab = {}
vocab["<pad>"] = 0
vocab["<unk>"] = 1
for index, token in enumerate(ordered_dict):
    vocab[token] = index

print([vocab[token] for token in ['this', 'is', 'an', 'example']])


def text_pipeline(x): return [vocab[token] for token in tokenizer(x)]
def label_pipeline(x): return 1. if x == 'pos' else 0.


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
                      shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_batch)
train_dl = DataLoader(test_dataset, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_batch)


embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
text_encoded_input = torch.LongTensor([[1,2,4,5],[4,3,2,0]])
print(embedding(text_encoded_input))