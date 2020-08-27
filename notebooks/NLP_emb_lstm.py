import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, GlobalAveragePooling1D
from sklearn_pipeline import *

class NLP_emb_lstm(object):
    def __init__(self, att, target, embedding_dim):
        self.att = att
        self.target = target
        model = Sequential() 
        model.add(Embedding(400, embedding_dim))
        model.add(LSTM(100)) 
#         model.add(GlobalAveragePooling1D())
        model.add(Dense(500, activation='relu')) 
        model.add(Dense(1, activation='sigmoid')) 
        
        print(model.summary())

        METRICS = [tf.metrics.BinaryAccuracy(name='ACCURACY'), tf.metrics.BinaryAccuracy(name='PRECISION'), tf.metrics.BinaryAccuracy(name='RECALL'), \
                  tf.metrics.BinaryAccuracy(name='F1Score')]
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=METRICS) 

        self.model = model
    
    def fit(self, size_batch, no_epoch):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        history_augmented = self.model.fit(self.att, self.target, verbose=1, batch_size = size_batch, epochs = no_epoch, validation_split=0.2,
                   shuffle=True, callbacks=[tensorboard_callback])
        
        return history_augmented


if __name__ == "__main__":
    train_df=pd.read_csv('../data/processed_train.csv')
    test_df=pd.read_csv('../data/processed_test.csv')
    word_max_features=500
    keyword_max_features=100
    location_max_features=100
    url_max_features=100
    embedding_dim = 128
    train_X_augmented, test_X_augmented, train_target = train_test_augmented(train_df, \
        test_df, word_max_features, keyword_max_features, location_max_features, \
            url_max_features) 

    embedding_model =  NLP_emb_lstm(train_X_augmented, train_target.values, embedding_dim) 

    embedding_model.fit(16, 10)
