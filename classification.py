import pandas
import numpy
import sklearn.datasets
from decision_tree import decision_tree
from collections import Counter
import sklearn
import json
from itertools import combinations, chain
from abc import ABC, abstractmethod

# data = pandas.read_excel("./sampledataClassification.xlsx")
# attributes = [attribute for attribute in data.columns]

'''
Classification Methods
'''


class ClassificationMethod(ABC):
    @abstractmethod
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, features, X_train, Y_train, label_names):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

# Base class template for decision tree classification


class ClassificationDecisionTree(ClassificationMethod):
    @abstractmethod
    def attribute_selection(features, data_partition, labels, classifiers):
        pass

    @abstractmethod
    def set_model(self, model):
        self.model = model

# Classification using NaiveBayes


class ClassificationNaiveBayes(ClassificationMethod):
    def __init__(self):
        super().__init__()
        self.priori = dict()
        self.posteriori = dict()
        self.class_counts = dict()

    def train(self, features, X_train, Y_train):
        """   
        Compute all posteriori probability from the training set
        Args:
            features (np.1darray): vector of description of data
            X_train (np.ndarray): set of training tuple 
            Y_train (np.1darray): labels of the training tuples
        """
        n = len(Y_train)
        self.class_counts = {key: 0 for key in set(Y_train)}

        for label in Y_train:
            self.class_counts[label] += 1

        # P(c_i) for all class label c_i
        for c_i, value in self.class_counts.items():
            self.priori[c_i] = value / n

        # tuple1 ['attr1', 'attr2', 'attr3', ... ]
        # tuple2 ['attr1', 'attr2', 'attr3', ... ]
        # tuple3 ['attr1', 'attr2', 'attr3', ... ]
        #            :
        #            :
        for attr_ind in range(len(features)):
            # Try treating the attribute as continuous
            try:
                values = [float(i) for i in X_train[:, attr_ind]]
                self.posteriori[attr_ind] = {"type": 0}
                for classifier in self.class_counts.keys():
                    self.posteriori[attr_ind][classifier] = {}
                    values = [float(train_tuple[attr_ind]) for i, train_tuple in enumerate(
                        X_train) if Y_train[i] == classifier]
                    self.posteriori[attr_ind][classifier] = (
                        numpy.mean(values), numpy.std(values, ddof=0))
            # Discrete / Nominal Errors
            except ValueError:
                values = set([i for i in X_train[:, attr_ind]])
                self.posteriori[attr_ind] = {"type": 1}
                for value in values:
                    self.posteriori[attr_ind][value] = {}
                    for classifier in self.class_counts.keys():
                        # count the number of training tuples with x_k and belonging to classifier
                        count = sum([1 for i, tuple in enumerate(X_train)
                                     if tuple[attr_ind] == value and Y_train[i] == classifier])
                        # P(x_k | C_i) = |X_k and C_i| / |C_i|     , while simultaneously implementing Laplacian correction
                        self.posteriori[attr_ind][value][classifier] = (
                            count + 1) / (self.class_counts[classifier] + len(values))

    def predict(self, X_test):
        """
        Predict the label for X_test using the trained model
        """
        res = 0
        prediction = None
        for Ci in self.class_counts.keys():
            prob = 1
            # Compute probability assuming posteriori probability independence
            for ind, attr_value in enumerate(X_test):

                # Discrete attribute
                if self.posteriori[ind]["type"] == 1:
                    prob *= self.posteriori[ind][attr_value][Ci]

                # Continous attribute
                else:
                    mean, std_dev = self.posteriori[ind][Ci]
                    prob *= ClassificationNaiveBayes.gaussian(
                        attr_value, mean, std_dev)
            # Get the label with the highest probability
            if prob > res:
                res = prob
                prediction = Ci
        return prediction

    def gaussian(x, u, sigma):
        if sigma == 0:
            sigma = 1e-6
        p1 = 1/(numpy.sqrt(2*numpy.pi)*sigma)
        p2 = numpy.exp(-((x-u) ** 2) / (2*sigma ** 2))
        return p1*p2

# Decision tree classification using GINI index for attribute selection


class ClassificationGINI(ClassificationDecisionTree):
    def __init__(self):
        super().__init__()

    def set_model(self, model):
        return super().set_model(model)

    def attribute_selection(features, data_partition, labels, classifiers):
        """
        Decide the best attribute for a split by minimizing GINI index of a partition resulting from a split on an attribute
        """
        best_A = -1
        curr_best_score = numpy.inf
        curr_best_splitting_criterion = None
        for attribute_index in range(len(features)):
            # Missing values
            if (features[attribute_index] == None):
                continue
            try:  # Continuous values
                attribute_values = [float(x[attribute_index])
                                    for x in data_partition]

                # Sort data_partition in asc. based on values of column <attribute_index>
                sorted_indices = sorted(
                    range(len(attribute_values)), key=lambda x: data_partition[x][attribute_index])

                # Test midpoint between each pair of adjacent values
                score_A = numpy.inf
                split_point = 0
                for i in range(0, len(sorted_indices) - 1):
                    L1 = [labels[k] for k in sorted_indices[:i+1]]
                    L2 = [labels[k] for k in sorted_indices[i+1:]]

                    # Compute score of a partition, take note of it if its less than current best split
                    s = ClassificationGINI.gini_index_A(L1, L2, classifiers)
                    if s < score_A:
                        score_A = s
                        split_point = (
                            attribute_values[sorted_indices[i]]+attribute_values[sorted_indices[i+1]])/2
                if (curr_best_score > score_A):
                    best_A = attribute_index
                    curr_best_score = score_A
                    curr_best_splitting_criterion = split_point
            except ValueError:  # Discrete values
                attribute_values = [x[attribute_index]
                                    for x in data_partition]
                attribute_subsets = list(chain.from_iterable(combinations(
                    set(attribute_values), r) for r in range(1, len(set(attribute_values)))))

                score_A = numpy.inf
                split_value = None
                for subset in attribute_subsets:
                    # Tuples with attribute A's value in subset
                    L1 = [
                        labels[i] for (i, x) in enumerate(data_partition) if x[attribute_index] in subset]

                    # Tuples without attribute A's value in subset
                    L2 = [
                        labels[i] for (i, x) in enumerate(data_partition) if x[attribute_index] not in subset]

                    # Compute score of a partition, take note of it if its less than current best split
                    s = ClassificationGINI.gini_index_A(L1, L2, classifiers)
                    if s < score_A:
                        score_A = s
                        split_value = subset
                if curr_best_score > score_A:
                    best_A = attribute_index
                    curr_best_score = score_A
                    curr_best_splitting_criterion = split_value
        return best_A, curr_best_splitting_criterion

    def train(self, features, X_train, Y_train, label_names, subfeature_size=-1):
        c = set(Y_train)

        # Check if partition is 'cleaned' with only 1 class label
        if (len(c) == 1):
            ind = c.pop()
            D_tree = decision_tree()
            D_tree.assign_class_label(ind.item())
            D_tree.change_to_leaf_node()
            return D_tree

        # No features left to check
        if len(features) == 0:
            D_tree = decision_tree()
            label = Counter(Y_train)
            D_tree.assign_class_label(label.most_common(1)[0][0])
            D_tree.change_to_leaf_node()
            return D_tree

        # For Random Forest Ensemble method
        if (subfeature_size > 0):
            subfeature_size = min(subfeature_size, len(features))
            indices = numpy.random.choice(
                range(len(features)), size=subfeature_size, replace=False)
            # Choose best among k features to branch tree
            # This does not modify actual object's feature structure, rather to choose among the
            # subset of attributes for the best branching attribute
            subf = [features[i]
                    if i in indices else None for i in range(len(features))]
            (best_split_attribute_ind, split_criterion) = ClassificationGINI.attribute_selection(
                subf, X_train, Y_train, label_names)

        # Regular method, finding best attribute to split
        else:
            (best_split_attribute_ind, split_criterion) = ClassificationGINI.attribute_selection(
                features, X_train, Y_train, label_names)

        if (split_criterion == None or best_split_attribute_ind == -1):
            D_tree = decision_tree()
            label = Counter(Y_train)
            D_tree.assign_class_label(label.most_common(1)[0][0])
            D_tree.change_to_leaf_node()
            return D_tree

        D_tree = decision_tree()
        D_tree.assign_node_name(features[best_split_attribute_ind])
        D_tree.assign_node_index(best_split_attribute_ind)
        match split_criterion:
            case float():           # Splitting point method (continous-value attribute was chosen)
                D_tree.initialize_criterias(["D1", "D2"])
                D_tree.splitpoint = split_criterion

                # Index of tuples in partition that <= split_criterion
                indices = [i for i, tpl in enumerate(X_train) if tpl[best_split_attribute_ind]
                           <= split_criterion]

                # Left partition, D1 all tuples with attribute values <= split_point
                D1 = X_train[indices]
                L1 = Y_train[indices]

                # Right partition, D2 all tuples with attribute values > split_point
                D2 = numpy.delete(X_train, indices, axis=0)
                L2 = numpy.delete(Y_train, indices, axis=0)

                # If partitioning result in an empty subpartition, then take majority vote, turn the current node into a leaf, and assign the majority class label
                if (len(D1) == 0 or len(D2) == 0):
                    D_tree.change_to_leaf_node()
                    if (len(D1) == 0):
                        label = Counter(L2)
                        D_tree.assign_class_label(
                            label_names[label.most_common(1)[0][0]])
                    else:
                        label = Counter(L1)
                        D_tree.assign_class_label(
                            label_names[label.most_common(1)[0][0]])

                # Otherwise, build conditional branches and the subtrees associated
                else:
                    D_tree.add_child(
                        "D1", self.train(features, D1, L1, label_names, subfeature_size=subfeature_size))
                    D_tree.add_child(
                        "D2", self.train(features, D2, L2, label_names, subfeature_size=subfeature_size))
                return D_tree

            # Splitting attribute is nominal
            case tuple():
                D_tree.initialize_criterias(["D1", "D2"])
                D_tree.splitpoint = split_criterion
                indices = [i for i, tpl in enumerate(
                    X_train) if tpl[best_split_attribute_ind] in split_criterion]

                # Partition of data with split attribute's values in the split_criterion
                D1 = numpy.delete(X_train[indices],
                                  best_split_attribute_ind, axis=1)
                L1 = Y_train[indices]

                # Partitions of data with split attribute's values not in the split_criterion
                D2 = numpy.delete(numpy.delete(X_train, indices,
                                               axis=0), best_split_attribute_ind, axis=1)

                L2 = numpy.delete(Y_train, indices, axis=0)

                if (len(D1) == 0 or len(D2) == 0):
                    D_tree.change_to_leaf_node()
                    if (len(D1) == 0):
                        label = Counter(L2)
                        D_tree.assign_class_label(
                            label_names[label.most_common(1)[0][0]])
                    else:
                        label = Counter(L1)
                        D_tree.assign_class_label(
                            label_names[label.most_common(1)[0][0]])

                else:
                    # Remove the attribute from recursive consideration
                    features.pop((best_split_attribute_ind))
                    D_tree.add_child("D1", self.train(
                        features.copy(), D1, L1, label_names, subfeature_size=subfeature_size))
                    D_tree.add_child("D2", self.train(
                        features.copy(), D1, L1, label_names, subfeature_size=subfeature_size))
                return D_tree

    def predict(self, features, X_test: numpy.array):
        """
        Use trained GINI-model to predict the label of X_test
        """
        node = self.model
        while (not node.is_leaf()):
            match node.splitpoint:
                # If splitting attribute is nominal
                case tuple():
                    attribute = features.index(node.get_node_name())
                    subset = node.splitpoint
                    if (X_test[attribute] in subset):
                        node = node.get_criterias()["D1"]
                    else:
                        node = node.get_criterias()["D2"]

                # If splitting attribute is continuous
                case float():
                    split_point = node.splitpoint
                    attribute = node.get_node_index()
                    if (X_test[attribute] <= split_point):
                        node = node.get_criterias()["D1"]
                    else:
                        node = node.get_criterias()["D2"]
        prediction = node.get_class_label()
        return prediction

    def gini_index(data_labels, classifiers):
        """
        Iterate over a partition D to compute the probability that a tuple belongs to label C_m in D
        """
        score = 1
        for m in range(len(classifiers)):
            score -= numpy.power(len([i for i in range(len(data_labels))
                                      if data_labels[i] == classifiers[m]])/len(data_labels), 2)
        return score

    def gini_index_A(labels1, labels2, classifiers):
        """
        Compute GINI score of a partition
        """
        D = len(labels1) + len(labels2)

        score = (len(labels1) / D) * ClassificationGINI.gini_index(labels1, classifiers) + \
            (len(labels2) / D) * ClassificationGINI.gini_index(labels2, classifiers)
        return score


# Decision tree classification using ID3 entropy for attribute selection
class ClassificationID3(ClassificationDecisionTree):
    def __init__(self):
        super().__init__()

    def set_model(self, model):
        return super().set_model(model)

    def attribute_selection(features, data_partitions, labels, classifier):
        """
        Decide the best attribute for a split by maximize entropy of a partition resulting from a split on an attribute
        """
        best_A = -1
        curr_best_score = numpy.inf
        curr_best_splitting_criterion = None
        for attribute_index in range(len(features)):
            try:  # Continuous values
                attribute_values = [float(x[attribute_index])
                                    for x in data_partitions]
                # Sort data_partition in asc. based on values of column <attribute_index>
                sorted_indices = sorted(
                    range(len(attribute_values)), key=lambda x: data_partitions[x][attribute_index])

                # Test midpoint between each pair of adjacent attribute values
                score_A = numpy.inf
                split_point = 0

                # Check entropy score of the partition off each pair of adjacent attribute values
                for i in range(0, len(sorted_indices) - 1):
                    L1 = [labels[k] for k in sorted_indices[:i+1]]
                    L2 = [labels[k] for k in sorted_indices[i+1:]]
                    s = ClassificationID3.ID3_InfoAttribute(
                        [L1, L2], classifier)
                    if s < score_A:
                        score_A = s
                        split_point = (
                            attribute_values[sorted_indices[i]]+attribute_values[sorted_indices[i+1]])/2
                if (curr_best_score > score_A):
                    best_A = attribute_index
                    curr_best_score = score_A
                    curr_best_splitting_criterion = split_point
            except ValueError:  # Discrete values
                attribute_values = set([x[attribute_index]
                                        for x in data_partitions])
                score_A = numpy.inf
                partitions = []

                # Perform a split on the attribute, partitions into how ever many unique values of the splitting attribute
                for attr_value in attribute_values:
                    # Tuples with attribute A's value in subset
                    partitions.append([labels[i] for (i, x) in enumerate(
                        data_partitions) if x[attribute_index] == attr_value])

                # Compute ID3 score on the partition
                score_A = ClassificationID3.ID3_InfoAttribute(
                    partitions, classifier)
                if curr_best_score > score_A:
                    best_A = attribute_index
                    curr_best_score = score_A
                    curr_best_splitting_criterion = attribute_values
        return best_A, curr_best_splitting_criterion

    def train(self, features, X_train, Y_train, label_names):
        c = set(Y_train)
        # Base case, all tuples in D are of the same class label
        if len(c) == 1:
            ind = c.pop()
            D_tree = decision_tree()
            D_tree.assign_class_label(ind.item())
            D_tree.change_to_leaf_node()
            return D_tree

        # No more features to test
        if len(features) < 1:
            D_tree = decision_tree()
            label = Counter(Y_train)
            D_tree.assign_class_label(label.most_common(1)[0][0])
            D_tree.change_to_leaf_node()
            return D_tree

        # Choose which attribute to branch off
        (best_split_attribute_ind, split_criterion) = ClassificationID3.attribute_selection(
            features, X_train, Y_train, label_names)

        if (split_criterion == None or best_split_attribute_ind == -1):
            D_tree = decision_tree()
            label = Counter(Y_train)
            D_tree.assign_class_label(label.most_common(1)[0][0])
            D_tree.change_to_leaf_node()
            return D_tree

        D_tree = decision_tree()
        D_tree.assign_node_name(features[best_split_attribute_ind])
        D_tree.assign_node_index(best_split_attribute_ind)

        match split_criterion:
            case float():           # Splitting point method (continous-value attribute was chosen)
                D_tree.initialize_criterias(["D1", "D2"])
                D_tree.splitpoint = split_criterion

                # Index of tuples in partition that <= split_criterion
                indices = [i for i, tpl in enumerate(X_train) if tpl[best_split_attribute_ind]
                           <= split_criterion]

                # Left partition, D1 all tuples with attribute values < split_point
                D1 = X_train[indices]
                L1 = Y_train[indices]

                # Right partition, D2 all tuples with attribute values < split_point
                D2 = numpy.delete(X_train, indices, axis=0)
                L2 = numpy.delete(Y_train, indices, axis=0)

                # If partitioning result in an empty subpartition, then take majority vote, turn the current node into a leaf, and assign the majority class label
                if (len(D1) == 0 or len(D2) == 0):
                    D_tree.change_to_leaf_node()
                    if (len(D1) == 0):
                        label = Counter(L2)
                        D_tree.assign_class_label(
                            label_names[label.most_common(1)[0][0]])
                    else:
                        label = Counter(L1)
                        D_tree.assign_class_label(
                            label_names[label.most_common(1)[0][0]])

                # Otherwise, build conditional branches and the subtrees associated
                else:
                    D_tree.add_child(
                        "D1", self.train(features.copy(), D1, L1, label_names))
                    D_tree.add_child(
                        "D2", self.train(features.copy(), D2, L2, label_names))
                return D_tree

            # Discrete attribute is chosen
            case tuple():
                D_tree.initialize_criterias([split_criterion])
                D_tree.splitpoint = split_criterion
                for criterion in split_criterion:
                    indices = [i for i, tpl in enumerate(
                        X_train) if tpl[best_split_attribute_ind] == criterion]

                    # Partition of data with splitting attribute's values in the split_criterion, while removing the splitting attribute column from further consideration
                    D1 = numpy.delete(
                        X_train[indices], best_split_attribute_ind, axis=1)
                    L1 = Y_train[indices]

                    features.pop((best_split_attribute_ind))
                    D_tree.add_child(criterion, self.train(
                        features.copy(), D1, L1, label_names))
                return D_tree

    def predict(self, features, X_test: numpy.array):
        """
        Predict the label of X_test using the trained model
        """
        node = self.model
        while (not node.is_leaf()):
            match node.splitpoint:
                case float():
                    split_point = node.splitpoint
                    attribute = node.get_node_index()
                    if (X_test[attribute] <= split_point):
                        node = node.get_criterias()["D1"]
                    else:
                        node = node.get_criterias()["D2"]
                case tuple():
                    attribute = features.index(node.get_node_name())
                    node.get_criterias()[X_test[attribute]]
        prediction = node.get_class_label()
        return prediction

    def ID3_entropy(data_labels, classifiers):
        """
        Compute the 'information needed' to classify tuples in current partition D
        """
        r = 0
        for ind, classifier in enumerate(classifiers):
            p = len([i for i in range(len(data_labels))
                     if data_labels[i] == classifier]) / len(data_labels)
            if p == 0:
                continue
            r += p * numpy.log2(p)
        return -1*r

    def ID3_InfoAttribute(partitions, classifiers):
        """
        Compute the 'information needed' to classify tuples, supposed we split on attribute A
        """
        D = sum([len(x) for x in partitions])
        score = 0
        for partition in partitions:
            score += (len(partition) / D) * \
                ClassificationID3.ID3_entropy(partition, classifiers)
        return score


# Helper functions
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


def main():
    (X_train, Y_train, X_test, Y_test, features,
     classifier_names) = data_preprocessing(n=10)

    # -------------------------------------------------------------------------------------
    # prediction = classify(features, X_test, tree)
    # strings = iterate_through_tree(tree)
    # cleaned_data = clean_tree_output(strings)

    gini = ClassificationGINI()
    gini.set_model(gini.train(features.copy(),
                   X_train, Y_train, classifier_names))

    id3 = ClassificationID3()
    id3.set_model(id3.train(features.copy(), X_train,
                  Y_train, classifier_names))

    NBayes = ClassificationNaiveBayes()
    NBayes.train(features, X_train, Y_train)

    # model = build_classification_naive_bayes_model(
    #    features, X_train, Y_train)
    GINI_model_predictions = [gini.predict(features, x) for x in X_test]
    ID3_model_predictions = [id3.predict(features, x) for x in X_test]
    NBayes_predictions = [NBayes.predict(x) for x in X_test]

    print(compare(GINI_model_predictions, Y_test))
    print(compare(ID3_model_predictions, Y_test))
    print(compare(NBayes_predictions, Y_test))


if __name__ == "__main__":
    main()
