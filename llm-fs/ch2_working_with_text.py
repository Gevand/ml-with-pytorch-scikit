from importlib_metadata import version
import tiktoken
import re

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
