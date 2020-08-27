import numpy as np

def boosting_parameters():

    parameters = {
        "loss":["deviance"],
        "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
#         "min_samples_split": np.linspace(0.1, 0.5, 12),
#         "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth":[3,5, 10, 20],
        "max_features":["log2","sqrt"],
        "criterion": ["friedman_mse",  "mae"],
        "subsample":[0.5, 0.75, 0.9],
        "n_estimators":[10, 30, 100, 300, 1000]
        }
    
    return parameters