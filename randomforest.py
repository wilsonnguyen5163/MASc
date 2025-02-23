from collections import Counter
import helpers
import numpy
import classification
import sklearn


class RandomForest:
    def __init__(self, k, f_subset_size: int):
        self.partitioning_method = None
        self.num_trees = k
        self.trees = []
        self.f_size = f_subset_size

    def set_partitioning_method(self, method: classification.ClassificationDecisionTree):
        self.partitioning_method = method

    def train(self, X_train, Y_train, features, classifier_names):
        for i in range(self.num_trees):
            indices = helpers.bootstrap_sampling(X_train)
            Xi = X_train[indices]
            Yi = Y_train[indices]
            Ti = classification.ClassificationGINI()

            # Train tree Ti using bootstrapped training set, specifying number of random
            Ti.set_model(Ti.train(features, Xi, Yi, classifier_names,
                                  subfeature_size=self.f_size))
            self.trees.append(Ti)

    def predict(self, features, X_test):
        res = Counter([tree.predict(features, X_test) for tree in self.trees])
        return res.most_common(1)[0][0]

    def individual_prediction(self, features, X_test):
        return [tree.predict(features, X_test) for tree in self.trees]


def data_preprocessing(n=5):
    (data, labels) = sklearn.datasets.load_breast_cancer(
        return_X_y=True, as_frame=True)
    features = data.columns.tolist()
    classifier_names = list(set(labels))

    # Use pandas to sample without replacement the data to be used for testing
    X_test = data.sample(frac=1/n, replace=False, random_state=30)

    # Partition dataset into test and training sets, using holdout method
    Y_test = labels.loc[X_test.index]
    X_train = data.drop(X_test.index)
    Y_train = labels.drop(X_test.index)

    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy(), features, classifier_names


def main():

    (X_train, Y_train, X_test, Y_test, features,
     classifier_names) = data_preprocessing(n=5)

    # Hyper parameters
    num_trees = 10
    subfeature_size = int(numpy.ceil(numpy.sqrt(len(features))))
    print(subfeature_size)

    RndForest = RandomForest(num_trees,  subfeature_size)
    RndForest.train(X_train, Y_train, features, classifier_names)
    predictions = [RndForest.predict(features, x) for x in X_test]

    gini = classification.ClassificationGINI()
    gini.set_model(gini.train(features, X_train, Y_train, classifier_names))

    # single_tree_prediction = [gini.predict(features, x) for x in X_test]

    print(classification.compare(predictions, Y_test))


if __name__ == "__main__":
    main()
