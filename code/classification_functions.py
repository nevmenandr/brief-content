import numpy as np
import pandas as pd

from smart_open import smart_open
from numpy import random
import gensim
import nltk
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from gensim.models.fasttext import FastText
from nltk.corpus import stopwords


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

    
def evaluate_prediction(predictions, target, tags, title="Confusion matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, tags, title + ' (normalized)')

    
def predict(vectorizer, classifier, data, target, tags):
    data_features = vectorizer.transform(data)
    predictions = classifier.predict(data_features)
    evaluate_prediction(predictions, target, tags)

    
def most_influential_words(vectorizer, logreg, genre_index=0, num_words=10):
    features = vectorizer.get_feature_names()
    max_coef = sorted(enumerate(logreg.coef_[genre_index]), key=lambda x:x[1], reverse=True)
    return [features[x[0]] for x in max_coef[:num_words]]


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
