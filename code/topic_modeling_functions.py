import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def get_data_vectorized(vect, data):
    data = vect.fit_transform(data)
    feature_names = vect.get_feature_names()
    
    return data, feature_names

def get_models_for_n_topics(model_name, data, start, stop, step):
    res = []
    if model_name == 'nmf':
        stop = min(data.shape[0] + 1, stop)
    
    for i in range(start, stop, step): 
        if model_name == 'lda':
            model = LatentDirichletAllocation(n_components=i, max_iter=5, learning_method='online',
                                            learning_offset=50., random_state=42).fit(data)
        elif model_name == 'nmf':
            model = NMF(n_components=i, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd').fit(data)
        
        res.append(model.components_)
    return res

def get_n_top_words(models, feature_names, n_top_words):
    res = []
    for model in models:
        for topic_idx, topic in enumerate(model):
            res.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return res

def get_index(start, stop, step):
    index_1 = []
    index_2 = []
    for i in range(start, stop, step):
        index_1.extend(['n_topics_' + str(i)] * i)
        index_2.extend(['n_topic_' + str(j) for j in range(1, i+1)])
    
    arrays = [np.array(index_1), np.array(index_2)]
    return arrays
    
def get_words_for_topics(data, count_vect, tfidf_vect, start, stop, step, n_top_words):
    count_data, count_feature_names = get_data_vectorized(count_vect, data)
    tfidf_data, tfidf_feature_names = get_data_vectorized(tfidf_vect, data)
    
    lda_res = get_models_for_n_topics('lda', count_data, start, stop, step)
    nmf_res = get_models_for_n_topics('nmf', tfidf_data, start, stop, step)
    
    lda_topics = get_n_top_words(lda_res, count_feature_names, n_top_words)
    nmf_topics = get_n_top_words(nmf_res, tfidf_feature_names, n_top_words)
    
    df_results = pd.DataFrame({'lda': lda_topics}, index=get_index(start, stop, step))
    df_results['nmf'] = pd.Series(nmf_topics, index=get_index(start, min(tfidf_data.shape[0] + 1, stop), step))

    return df_results