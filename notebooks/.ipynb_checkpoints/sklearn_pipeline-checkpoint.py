import pandas as pd
from list_to_corpus import *
from helper import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def train_test_augmented(train_df, test_df, word_max_features, keyword_max_features, location_max_features, url_max_features):
    # train_df=pd.read_csv('../data/processed_train.csv')
    # test_df=pd.read_csv('../data/processed_test.csv')
    train_df['text']=train_df.text.fillna('uns_text')
    test_df['text']=test_df.text.fillna('uns_text')
    train_corpus=list_corpus(train_df['text'])
    test_corpus=list_corpus(test_df['text'])
    #extract target
    train_target=train_df.pop('target')
    #text_vectorization
    vectorizer = TfidfVectorizer(max_features=word_max_features)
    train_tf_X = vectorizer.fit_transform(train_corpus)
    test_tf_X = vectorizer.transform(test_corpus)
    #modified_keyword
    vectorizer = CountVectorizer(max_features=keyword_max_features, binary=True)
    train_dummy_keyword = vectorizer.fit_transform(list_corpus(train_df['modified_keyword'])).toarray()
    test_dummy_keyword = vectorizer.transform(list_corpus(test_df['modified_keyword'])).toarray()
    #modified_location
    vectorizer = CountVectorizer(max_features=location_max_features, binary=True)
    train_dummy_location = vectorizer.fit_transform(list_corpus(train_df['modified_location'])).toarray()
    test_dummy_location = vectorizer.transform(list_corpus(test_df['modified_location'])).toarray()
    #modified_urls
    vectorizer = CountVectorizer(max_features=url_max_features, binary=True)
    train_dummy_urls = vectorizer.fit_transform(list_corpus(train_df['urls'])).toarray()
    test_dummy_urls = vectorizer.transform(list_corpus(test_df['urls'])).toarray()
    #augemntation
    train_X_augmented = np.concatenate((train_tf_X.toarray(), train_dummy_location, train_dummy_keyword, train_dummy_urls, \
                                        train_df.url_count.values.reshape(-1,1),train_df.emoji_count.values.reshape(-1,1)), axis=1)
    test_X_augmented = np.concatenate((test_tf_X.toarray(), test_dummy_location, test_dummy_keyword, test_dummy_urls, \
                                        test_df.url_count.values.reshape(-1,1), test_df.emoji_count.values.reshape(-1,1)), axis=1)

    return train_X_augmented, test_X_augmented, train_target