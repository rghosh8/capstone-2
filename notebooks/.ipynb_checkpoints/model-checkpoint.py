import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, GlobalAveragePooling1D

def embedding(embedding_dim, max_features):
    
    model = Sequential() 
    model.add(Embedding(max_features + 1, embedding_dim))
    model.add(LSTM(100)) 
#     model.add(GlobalAveragePooling1D())
    model.add(Dense(1)) 
    
    METRICS = [tf.metrics.BinaryAccuracy(name='ACCURACY')]
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=METRICS) 
    
    return model

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
 