import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def list_element(lst):
    if len(lst)==0:
        return np.NaN
    else:
        return lst[0]
    
def tfidf_vectorizer(corpus, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    feature_name = vectorizer.get_feature_names()
    
    return X.todense(), feature_name
 