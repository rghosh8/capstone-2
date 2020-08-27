import numpy as np

def boosting_parameters():

    parameters = {
#       "loss":["deviance"],
        "learning_rate": [0.001, 0.01, 0.1],
        "subsample":[0.5, 0.75],
        "n_estimators":[300, 900, 4500]
    }
    
    return parameters