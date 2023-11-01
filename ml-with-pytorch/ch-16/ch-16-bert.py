import gzip
import shutil
import time
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import transformers
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = 'cuda'
NUM_EPOCHS = 3

df = pd.read_csv('ch-16/movie_data.csv')
print(df.head(5))

train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values

valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values

test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.length = len(self.labels)

    def __getitem__(self, idx):
        if idx >= self.length:
            raise StopIteration

        try:
            item = {key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        except:
            raise StopIteration

    def __len__(self):
        return len(self.labels)


train_dataset = IMDBDataset(train_encodings, train_labels)
test_dataset = IMDBDataset(test_encodings, test_labels)
valid_dataset = IMDBDataset(valid_encodings, valid_labels)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False)

model.to(DEVICE)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            # Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
        return correct_pred.float()/num_examples * 100


start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']

        optim.zero_grad()
        loss.backward()
        optim.step()

        if not batch_idx % 250:
            print(f'Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d}'
                  f' | Batch '
                  f'{batch_idx:04d}/'
                  f'{len(train_loader):04d} | '
                  f'Loss: {loss:.4f}')

    model.eval()
    with torch.set_grad_enabled(False):
        print(f'Training accuracy: '
              f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
              f'\nValid accuracy: '
              f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')
        print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')
