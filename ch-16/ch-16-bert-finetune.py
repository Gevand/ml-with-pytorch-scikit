import numpy as np
from datasets import load_metric
import gzip
import shutil
import time
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import Trainer, DistilBertForSequenceClassification, TrainingArguments, DistilBertTokenizerFast

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

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased')
model.to(DEVICE)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='ch-16/results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='ch-16/logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optim, None)  
)

start_time = time.time()
trainer.train()

print(f'Total Training Time: 'f'{(time.time() - start_time)/60:.2f} min')
print(trainer.evaluate())

