import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(dataset):
    X = dataset.data
    y = dataset.target

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=1)

for random_state in [42, 1]:
    print("Random state:", random_state)
    for k in range(1, 11):
        knn = KNN(k=k, distance_metric='euclidean')
        if random_state == 42:
            knn.fit(X_train_1, y_train_1)
            predictions = knn.predict(X_test_1)
            accuracy = accuracy_score(y_test_1, predictions)
            print("Accuracy for k =", k, ":", accuracy)
        elif random_state == 1:
            knn.fit(X_train_2, y_train_2)
            predictions = knn.predict(X_test_2)
            accuracy = accuracy_score(y_test_2, predictions)
            print("Accuracy for k =", k, ":", accuracy)
