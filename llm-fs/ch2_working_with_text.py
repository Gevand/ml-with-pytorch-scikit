from importlib_metadata import version
import tiktoken
import re
import torch
from torch.utils.data import Dataset, DataLoader
with open("verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Total number of chars {len(raw_text)}")
print(raw_text[10:99])

text = "Hello, world. This, is a test"
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
pre_processed = [item.strip() for item in pre_processed if item.strip()]

print(pre_processed[:30])

all_words = sorted(list(set(pre_processed)))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50:
        break


class simple_tokenizer_v1:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, raw_text):
        pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        pre_processed = [item.strip()
                         for item in pre_processed if item.strip()]
        ids = [self.str_to_int[s] for s in pre_processed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = simple_tokenizer_v1(vocab=vocab)

text = """"It's the last he painted, you know," Mrs. Gisburn said"""
ids = tokenizer.encode(text)
print(ids)
undo_ids = tokenizer.decode(ids)
print(undo_ids)

pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
pre_processed = [item.strip() for item in pre_processed if item.strip()]
all_words = sorted(list(set(pre_processed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_words)}

print(list(vocab.items())[-5:])


class simple_tokenizer_v2:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, raw_text):
        pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        pre_processed = [item.strip()
                         for item in pre_processed if item.strip()]
        pre_processed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in pre_processed
        ]
        ids = [self.str_to_int[s] for s in pre_processed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
tokenizer = simple_tokenizer_v2(vocab=vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))


print("tiktoken version", version("tiktoken"))
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of some"

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode(integers))

print(tokenizer.encode("Akwirw ier"))

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1: context_size+1]
print(f"x: {x}")
# so it lines up, first number length + 3 for ,
print(f"y: {' ' * (len(str(x[0])) + 3)}{y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenier = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4,
                         max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    return dataloader


data_loader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(data_loader)
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)


vocab_size = 6
output_dim = 3
input_ids = torch.tensor([2, 3, 5, 1])
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))


output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
data_loader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length)
data_iter = iter(data_loader)
inputs, targets = next(data_iter)
print("Token IDs", inputs)
print("Inputs shape", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embeddings_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embeddings_layer(torch.arange(max_length))
print(pos_embeddings, pos_embeddings_layer.weight)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
