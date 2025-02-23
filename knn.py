import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# data = pd.read_csv("./knn_dataset_100.csv")
data = load_iris()
X = data.data
y = data.target


def partition_data(data, test_size):
    ratio = 1/test_size
    test_set = data[len(data)-round(ratio*len(data)):]
    training_set = data[:-round(ratio*len(data))]
    return (test_set, training_set)


def my_knn_classify(X_test, X_train, y_train, k=3):
    """
    Classify X_test based on k-nearest tuples' majority labels in X_train
    """
    classification = []
    for test_point in X_test:
        distances = []
        for index in range(X_train.shape[0]):
            d = [(test_point[a]-X_train[index][a])**2
                 for a in range(len(test_point))]
            distances.append(
                (np.sqrt(sum(d)), index))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        k_nearest_labels = [y_train[k[1]] for k in k_nearest]
        majority = np.bincount(k_nearest_labels).argmax()
        classification.append(majority)
    return classification


def tested_knn_classify(X_test, X_train, y_train, k=3):
    predictions = []
    for test_point in X_test:
        distances = []
        for training_point in X_train:
            distance = np.sqrt(
                sum([(test_point[i]-training_point[i])**2 for i in range(len(test_point))]))
            distances.append(distance)
        nearest_neighbours = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_neighbours]
        predictions.append(np.bincount(nearest_labels).argmax())
    return np.array(predictions)


def evaluate_quality(model: pd.DataFrame, true_val: pd.DataFrame):
    count = len(true_val)
    correct = 0
    for i, point in model.iterrows():
        correct = correct + \
            1 if model.loc[i]["Class"] == true_val.loc[i]["Class"] else correct

    print("\nClassification quality: ***************")
    print("Total count = {n} ---- Correct = {c} ------- Quality: {q}".format(
        n=count, c=correct, q=correct/count))
    return (correct/count)


def knn_classify(X_test, X_train, y_train, k=3, mode=1):
    match mode:
        case 1:  # My method
            return my_knn_classify(X_test, X_train, y_train, k)
        case 0:  # tested_method (benchmark)
            return tested_knn_classify(X_test, X_train, y_train, k)
        case _:
            pass


def main():

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    res = knn_classify(X_test, X_train, y_train, k=9, mode=1)

    # SKLearn KNN method
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)  # Set K=3 for comparison
    sklearn_knn.fit(X_train, y_train)
    sklearn_pred = sklearn_knn.predict(X_test)

    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    custom_accuracy = accuracy_score(y_test, res)

    print("sklearn accuracy: {a} ------- my_model accuracy: {b}".format(
        a=sklearn_accuracy, b=custom_accuracy))


if __name__ == "__main__":
    main()
