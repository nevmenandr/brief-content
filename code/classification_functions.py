import numpy as np
import pandas as pd
import gensim
import nltk
import os
import matplotlib.pylab as plt
from smart_open import smart_open
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from gensim.models.fasttext import FastText

import logging
logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def print_plot(df, index, label):
    example = df[df.index == index][['text', label]].values[0]
    if len(example) > 0:
        print(example[0], '\n\n')
        print(label + ':', example[1])
        
def plot_confusion_matrix(cm, tags, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest',  cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(tags))
    target_names = tags
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate_prediction(target, predictions, tags, title="Confusion matrix"):
    print(classification_report(target, predictions))
    print('accuracy ', accuracy_score(target, predictions))
    print('Макросредняя F1 мера - ', f1_score(target, predictions, average='macro'))
    print('Микросредняя F1 мера - ', f1_score(target, predictions, average='micro'))
    print()
    
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, tags, title + ' (normalized)')
    
def most_influential_words(vectorizer, logreg, genre_index=0, num_words=20):
    features = vectorizer.get_feature_names()
    max_coef = sorted(enumerate(logreg.coef_[genre_index]), key=lambda x:x[1], reverse=True)
    return [features[x[0]] for x in max_coef[:num_words]]

def save_wordlists(vectorizer, name, clf, labels):
    d = {}
    for i in range(len(labels)):
        d[labels[i]] = most_influential_words(vectorizer, clf, genre_index=i, num_words=20)
    df = pd.DataFrame(d)
    with open(name + '.txt', 'w', encoding='utf-8') as fw:
        df.to_csv(name + '.tsv', sep='\t', index=None)

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list ])
