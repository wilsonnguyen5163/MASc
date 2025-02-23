import numpy as np
import pandas
import sklearn
import matplotlib.pyplot as plt


def bootstrap_sampling(data):
    """
    Perform a sampling of data using bootstrap method.
    Return the data indices of the sample

    Args:
        data (np_array): dataset of objects
    """
    indices = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
    return indices


def euclidean_distance(p1, p2):
    if len(p1) != len(p2):
        return 0
    dist = 0
    for i in range(len(p1)):
        dist += np.pow(p1[i]-p2[i], 2)
    return np.sqrt(dist)


def data_preprocessing(n=5):
    (data, labels) = sklearn.datasets.load_wine(
        return_X_y=True, as_frame=True)
    features = data.columns.tolist()
    classifier_names = list(set(labels))

    # Use pandas to sample without replacement the data to be used for testing
    X_test = data.sample(frac=1/n, replace=False, random_state=1)

    # Partition dataset into test and training sets, using holdout method
    Y_test = labels.loc[X_test.index]
    X_train = data.drop(X_test.index)
    Y_train = labels.drop(X_test.index)

    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy(), features, classifier_names


def data_preprocessing_read_from_file(n=5):
    """Assumes the file has classifier on the right-most column

    Args:
        n (int, optional): _description_. Defaults to 5.
    """
    data = pandas.read_excel(
        "F:\\University\\MASc - Concordia\\Algorithms\\sampledataClassification.xlsx")
    features = data.columns.tolist()[:-1]
    classifier = data.columns.tolist()[-1]
    tuples = data.iloc[:, :-1]
    tuples_classifiers = data.iloc[:, -1:]
    classifier_names = list(
        set([j for i in tuples_classifiers.to_numpy() for j in i]))

    X_test = tuples.sample(frac=1/n, replace=False, random_state=1)
    Y_test = tuples_classifiers.loc[X_test.index]
    X_train = tuples.drop(X_test.index)
    Y_train = tuples_classifiers.drop(X_test.index)

    return X_train.to_numpy(), Y_train.to_numpy().ravel(), X_test.to_numpy(), Y_test.to_numpy().ravel(), features, classifier_names


def compare(pred, real):
    res = [1 for i in range(len(pred)) if pred[i] == real[i]]
    return sum(res)/len(real)
