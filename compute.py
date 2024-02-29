import numpy as np
from sklearn.model_selection import train_test_split

def load_data(dataset):
    X = dataset.data
    y = dataset.target
    return X, y

def split_data(X, y, test_size=0.2, random_state=52):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def evaluate_model(predictions, y_test):
    accuracy = np.mean(predictions == y_test)
    return accuracy
