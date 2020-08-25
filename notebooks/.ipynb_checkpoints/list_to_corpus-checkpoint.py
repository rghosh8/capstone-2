import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

def remove_urls(sent):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',sent)

def list_corpus(lst):
    '''
        converts a list of string into corpus
    '''
    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    punctuation_ = set(string.punctuation)

    corpus = []
    for item in lst:
        item = remove_urls(item)
        tokenized = [word_tokenize(c.lower()) for c in item.split()]
        docs = [[word for word in words if word not in stop and word not in punctuation_]
                for words in tokenized]
        docs=list(filter(lambda a: a != [], docs))

        string_ = ''
        for doc in docs:
            string_ += ' '+ doc[0]
        docs1 = [porter.stem(word) for word in string_.split(' ')]
        for doc1 in docs1:
            string_ += ' '+ doc1
        string_=string_[1:]
        corpus.append(string_)
        
    return corpus