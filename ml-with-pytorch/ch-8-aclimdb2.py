import numpy as np
import re
from nltk.corpus import stopwords
from typing import Callable, Any, Iterable, Tuple
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind

stop = stopwords.words('english')
basepath = '/mnt/d/Books/Python/ml-with-pytorch-scikit/aclImdb'
csv_path = basepath + "/movie_data.csv"


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)  # clean out all <html> in the text
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path: str, skip_header=True) -> Tuple[str, str]:
    with open(path, 'r', encoding='utf-8') as csv:
        if skip_header:
            next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream: Callable[[str, bool], Tuple[str, str]], size=1):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


print(next(stream_docs(csv_path)))
print(get_minibatch(stream_docs(csv_path), size=10))

vect = HashingVectorizer(decode_error='ignore', n_features=2 **
                         21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1)
doc_stream = stream_docs(path=csv_path)

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

# total samples is 50k, select the first 45k
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')
