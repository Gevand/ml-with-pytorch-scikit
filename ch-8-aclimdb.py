from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import re
import pyprind
import pandas as pd
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# dataset can be found at https://ai.stanford.edu/~amaas/data/sentiment/
np.random.seed(0)
basepath = '/mnt/d/Books/Python/ml-with-pytorch-scikit/aclImdb'

labels = {'pos': 1, 'neg': 0}
if not os.path.exists(basepath + '/movie_data.csv'):
    pbar = pyprind.ProgBar(50000, stream=sys.stdout)
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in sorted(os.listdir(path=path)):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()

    df.columns = ['review', 'sentiment']
    print(df.head())
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(basepath + '/movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv(basepath + '/movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})
print(df.head(3), df.shape)

count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet',
                'The sun in shining, the weather is sweet', 'and one and one is two'])
bag = count.fit_transform(docs)
print(bag)
print(count.vocabulary_)
print(bag.toarray())

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

print(df.loc[0, 'review'][-50:])


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


print(preprocessor(df.loc[0, 'review'][-50:]))
df['review'] = df['review'].apply(preprocessor)
porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# notice how thus becomes thu for this "stemming" algorithm
print(tokenizer_porter('runners like running and thus they run'))
nltk.download('stopwords')

stop = stopwords.words('english')
[w for w in tokenizer_porter(
    'a runner likes running and runs a lot') if w not in stop]

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf':[False],
        'vect__norm':[None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
]

lr_tfidf = Pipeline(
    [('vect', tfidf), ('clf', LogisticRegression(solver='liblinear'))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                           scoring='accuracy', cv=5, verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)
print(f'Best parameter set: {gs_lr_tfidf.best_params_}')
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')
# Best parameter set: {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': <function tokenizer at 0x7febddfaf790>}
# CV Accuracy: 0.897