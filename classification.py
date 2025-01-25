import sys
import pandas
import numpy
from decision_tree import decision_tree
import itertools

data = pandas.read_excel("./sampledataClassification.xlsx")
attributes = [attribute for attribute in data.columns]


def info_partition(data_partition, classifiers, classifier_attr_name):
    classifier_frequencies = {x: 0 for x in classifiers}
    data_partition = data_partition.reset_index(drop=True)

    for i in range(data_partition.shape[0]):    # Iterating through rows
        classifier_frequencies[data_partition.loc[i]
                               [classifier_attr_name]] += 1

    r = 0
    for classifier in classifiers:
        r += -(classifier_frequencies[classifier] / data_partition.shape[0]) * numpy.log2(
            classifier_frequencies[classifier] / data_partition.shape[0])
    return r


def info_attribute(data_partition, attr, attr_values, classifiers, classifiers_attr_name):
    attr_classifier_frequency = {
        x: {label: 0 for label in classifiers} for x in attr_values}
    data_size = data_partition.shape[0]
    r = 0
    data_partition = data_partition.reset_index(drop=True)
    for i in range(data_partition.shape[0]):
        row = data_partition.loc[i][:]
        attr_classifier_frequency[row[attr]
                                  ][row[classifiers_attr_name]] += 1

    for value in attr_values:
        s = sum([x for x in attr_classifier_frequency[value].values()])
        info_val = 0
        for label in classifiers:
            if (attr_classifier_frequency[value][label] == 0):
                continue
            info_val += -(attr_classifier_frequency[value][label] / s) * numpy.log2(
                attr_classifier_frequency[value][label] / s)
        r += (s/data_size) * info_val
    return r


def attribute_selection_ID3(attrs, data_partition, labels, classifier):
    info_D = info_partition(data_partition, labels, classifier)
    best = False
    best_val = -1
    for attr in attrs:
        attr_gain = info_D - \
            info_attribute(data_partition, attr,
                           attrs[attr], labels, classifier)
        if (attr_gain > best_val):
            best = attr
            best_val = attr_gain
    return best


def build_classification_decision_tree_model(attrs, partition, labels, classifier_name, method=1):
    match method:
        case 1:  # ID3
            # Base case, all tuples in D are of the same class label
            if len(set(partition.loc[:][classifier_name])) == 1:
                D_tree = decision_tree(classifier_name)
                label = next(iter(set(partition.loc[:][classifier_name])))
                D_tree.assign_class_label(label)
                D_tree.assign_class_attr(classifier_name)
                return D_tree
            if len(attrs) < 1:
                D_tree = decision_tree(classifier_name)
                majority = {}
                for entry in partition.loc[:][classifier_name]:
                    try:
                        majority[entry] += 1
                    except:
                        majority[entry] = 1

                label = max(majority, key=majority.get)
                D_tree.assign_class_label(label)
                D_tree.assign_class_attr(classifier_name)
                return D_tree

            # Choose which attribute to branch off
            attribute_branching = attribute_selection_ID3(
                attrs, partition, labels, classifier_name)
            node_branch_conditions = attrs[attribute_branching]

            D_tree = decision_tree(attribute_branching)
            D_tree.initialize_criterias(node_branch_conditions)

            attrs.pop(attribute_branching, None)
            # Recursively
            for condition in node_branch_conditions:
                D = partition.loc[partition[attribute_branching] == condition]
                partitioned_D = D.drop(columns=[attribute_branching])
                # print("Condition: {a}, ** current attribute: {b}\n".format(
                #    a=condition, b=attrs), partitioned_D, "\n")

                D_tree.add_child(condition, build_classification_decision_tree_model(
                    attrs.copy(), partitioned_D, labels, classifier_name))
            return D_tree
        case _:
            pass
    return


def training_data_initialize(classifier):
    classes = set(data.loc[:][classifier])
    attributes.remove(classifier)

    attributes_values = {attr: set(data.loc[:][attr]) for attr in attributes}
    return (attributes_values, classes)


def iterate_through_tree(tree: decision_tree, classifier):
    if tree.is_leaf():
        return "{c}?: {l}".format(c=tree.get_class_attr(), l=tree.get_class_label())

    strings = []
    for (condition, subtree) in tree.get_criterias().items():
        # print(condition, subtree)
        string = "{a}: ----- {cond} ------> {sub}".format(
            a=tree.to_string(), cond=condition, sub=iterate_through_tree(subtree, classifier))
        strings.append(string)

    return strings


def classify(data: pandas.DataFrame, trained_model: decision_tree, classifier):
    classified = {classifier: []}
    for i in range(data.shape[0]):
        tree = trained_model
        row = data.loc[i][:]
        branches = tree.get_criterias()
        while (len(branches) != 0):
            curr_node = tree.get_node_name()
            tree = branches[row[curr_node]]
            branches = tree.get_criterias()
        classified[classifier].append(tree.get_class_label())

    data[classifier] = classified[classifier]
    return data


def main():
    args = sys.argv[1:]  # TODO: Take arguments for data file for processing
    classifier = "buys_computer"
    (attribute_values, classes) = training_data_initialize(classifier)
    # print(data[data['age'] == "middle_age"])
    tree = build_classification_decision_tree_model(
        attribute_values.copy(), data, classes, classifier)

    for line in iterate_through_tree(tree, classifier):
        print(line)
    # classify_data = pandas.read_csv("./synthetic_test_data.csv")
    # print(classify(classify_data, tree, classifier))


if __name__ == "__main__":
    main()
