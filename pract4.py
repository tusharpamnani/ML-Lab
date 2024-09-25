import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=0.3, random_state=42)

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# KNN
def knn(X_train, y_train, test_point, k=3):
    distances = []
    # Calculate distance between test_point and all training points
    for i, train_point in enumerate(X_train):
        distance = euclidean_distance(test_point, train_point)
        distances.append((distance, y_train[i]))
    
    # Sort by distance and get the nearest k neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    
    # Get the labels of the k nearest neighbors and return the majority class
    labels = [label for _, label in k_nearest_neighbors]
    majority_vote = Counter(labels).most_common(1)[0][0]
    return majority_vote

# weighted knn
def weighted_knn(X_train, y_train, test_point, k=3):
    distances = []
    # Calculate distance between test_point and all training points
    for i, train_point in enumerate(X_train):
        distance = euclidean_distance(test_point, train_point)
        distances.append((distance, y_train[i]))
    
    # Sort by distance and get the nearest k neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    
    # Weighted voting: inversely proportional to distance
    weights = {}
    for distance, label in k_nearest_neighbors:
        if distance == 0:  # Avoid division by zero
            weight = float('inf')
        else:
            weight = 1 / distance
        if label in weights:
            weights[label] += weight
        else:
            weights[label] = weight

    # Return the class with the highest weighted vote
    return max(weights, key=weights.get)
def predict(X_train, y_train, X_test, k=3, weighted=False):
    predictions = []
    for test_point in X_test:
        if weighted:
            predictions.append(weighted_knn(X_train, y_train, test_point, k))
        else:
            predictions.append(knn(X_train, y_train, test_point, k))
    return predictions

def accuracy(y_true, y_pred):
    correct = sum(y1 == y2 for y1, y2 in zip(y_true, y_pred))
    return correct / len(y_true)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # KNN Prediction
    knn_predictions = predict(X_train, y_train, X_test, k=3, weighted=False)
    knn_acc = accuracy(y_test, knn_predictions)
    print(f"KNN Accuracy: {knn_acc * 100:.2f}%")

    # Weighted KNN Prediction
    weighted_knn_predictions = predict(X_train, y_train, X_test, k=3, weighted=True)
    weighted_knn_acc = accuracy(y_test, weighted_knn_predictions)
    print(f"Weighted KNN Accuracy: {weighted_knn_acc * 100:.2f}%")
