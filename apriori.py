import pandas
import sys
import numpy as np
from itertools import combinations, permutations


dataframe = pandas.read_excel("./sampledataApriori.xlsx")
rows = [tuple(dataframe.loc[i, "Item Bought"].replace(
    " ", "").split(",")) for i in range(dataframe.shape[0])]


# for row in rows:
#    print(set(['nuts']).issubset(set(row)), row)

minconf = .6
minsup = 3/9
totalEntries = dataframe.shape[0]


def initial_most_frequency():
    freq = dict()
    satisfyItem = dict()
    for rowNum in range(dataframe.shape[0]):
        items = dataframe.loc[rowNum, "Item Bought"].replace(
            " ", "").split(",")
        for item in items:
            if item not in freq:
                freq[item] = 1
            else:
                freq[item] += 1
    for item in freq:
        if (freq[item] / totalEntries >= minsup):
            satisfyItem[item] = freq[item]
    s = set([tuple([x]) for x in satisfyItem.keys()])
    return s


def superset(arr, size):
    """_summary_

    Args:
        arr (Set): set of values to be superseted
        size (int): size of individual set

    Returns:
        itemset (Set):
    """

    arr = set(tuple([x for xs in arr for x in xs]))
    # flatten set into singular (non-duplicate) list of elements
    itemset = set()
    for combo in list(combinations(arr, size)):
        flatten = tuple(combo)
        flatten = set(flatten)
        itemset.add(tuple(flatten))

    # Use frozenset to remove entry duplicates
    itemset = {frozenset(item) for item in itemset}
    itemset = [tuple(x) for x in itemset]
    return itemset


def frequency_analysis(arr, database):
    frequency = {}
    for combo in arr:
        t = tuple(combo)
        frequency[t] = 0
        for row in database:
            if set(combo).issubset(row):
                frequency[t] += 1
    return frequency


def supported_candidate(frequency):
    supported = set()
    for candididate, freq in frequency.items():
        if freq/totalEntries >= minsup:
            supported.add(candididate)
    return supported


def apriori_prune(selection):
    supported = set()
    curr_size = 2
    while (len(selection) > 0):
        freq = frequency_analysis(selection, rows)
        testing_set = supported_candidate(freq)
        if (len(testing_set) == 0):
            break
        supported = testing_set
        print("Apriori iteration: {i}  ---------- Current supported candidates: {l}".format(
            i=curr_size, l=supported))
        curr_size += 1
        selection = superset(supported, curr_size)
    return supported


def build_rules(candidate_set):
    rules = set()
    for candidate in candidate_set:
        combos = [x for xs in list(permutations(candidate, r)
                                   for r in range(2, len(candidate)+1))
                  for x in xs]
        for combo in combos:
            for i in range(1, len(combo)):
                rule = tuple([combo[:i], combo[i:]])
                rules.add(rule)
    return rules


def build_satisfying_rules(rules):
    satisfied = []
    for left, right in rules:
        conf = confidence(left, right)
        if (conf >= minconf):
            satisfied.append([left]+[right]+[conf])
    return satisfied


def confidence(left, right):
    left_and_right_frequency = frequency_analysis([left+right], rows)
    left_frequency = frequency_analysis([left], rows)
    return (left_and_right_frequency[(left+right)] / left_frequency[left])


def build_association_rules():
    init_selection = initial_most_frequency()
    print(
        "Initial Scanning: {init}\n*********************\n".format(init=init_selection))
    supported_items = apriori_prune(superset(init_selection, 2))

    print(
        "\n\nBuilding rules from supported items -------------{sel}-------------- min_conf {c}\n ".format(sel=supported_items, c=minconf))

    association_rules = build_satisfying_rules(build_rules(supported_items))
    # run functions to check the support of candidates
    display_rules(association_rules)
    return [x[:-1] for x in association_rules]


def display_rules(rules):
    for left, right, conf in rules:
        print("{l}   =>   {r},    confidence: {c}".format(
            l=list(left), r=list(right), c=conf))
    return


def main():
    args = sys.argv[1:]  # TODO: Take arguments for data file for processing
    build_association_rules()


if __name__ == "__main__":
    main()
