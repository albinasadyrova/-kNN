from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from distances import euclidean_distance, manhattan_distance, chebyshev_distance
from knn import KNN
from compute import load_data, split_data, evaluate_model

dataset = load_breast_cancer()
X, y = load_data(dataset)

X_train, X_test, y_train, y_test = split_data(X, y)

model = KNN(k=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test, euclidean_distance)

accuracy = evaluate_model(predictions, y_test)
print(f'Accuracy: {accuracy}')
