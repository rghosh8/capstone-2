import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, GlobalAveragePooling1D


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
    
    def fit(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        history_augmented = self.model.fit(self.att, self.target, verbose=1, batch_size = 16, epochs = 1, validation_split=0.2,
                   shuffle=True, callbacks=[tensorboard_callback])
        
        return history_augmented


    
#     model = 
#     model = tf.keras.Sequential([
#       layers.Embedding(max_features + 1, embedding_dim),
#       layers.Dropout(0.2),
#       layers.GlobalAveragePooling1D(),
#       layers.Dropout(0.2),
#       layers.Dense(1, activation='sigmoid')])

#     METRICS = [
#                 tf.metrics.BinaryAccuracy(name='ACCURACY')]
#                 tf.metrics.Precision(name='PRECISION'),
#                 tf.metrics.Recall(name='RECALL'),
#                 tf.metrics.AUC(name='AUC'),
#                 tf.metrics.TruePositives(name='TP'),
#                 tf.metrics.TrueNegatives(name='TN'),
#                 tf.metrics.FalsePositives(name='FP'),
#                 tf.metrics.FalseNegatives(name='FN')]

#     model.compile(loss='binary_crossentropy', 
#                   optimizer='adam', 
#                   metrics=METRICS)

#     return model
 