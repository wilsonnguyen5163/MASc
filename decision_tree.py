class decision_tree:
    __node_attribute = ""
    __children = []
    __splitting_criterias = {}
    __is_leaf = True
    __classification_label = ""
    __classification_attr = ""

    def __init__(self):
        self.__node_attribute = ""

    def __init__(self, name):
        self.__node_attribute = name

    def assign_node_name(self, name):
        self.__node_attribute = name

    def assign_class_label(self, label):
        self.__classification_label = label

    def assign_class_attr(self, attr):
        self.__classification_attr = attr

    def get_class_label(self):
        if (self.__is_leaf == True):
            return self.__classification_label
        return None

    def get_class_attr(self):
        if (self.__is_leaf == True):
            return self.__classification_attr
        return None

    def get_node_name(self):
        return self.__node_attribute

    def initialize_criterias(self, values):
        self.__splitting_criterias = {value: "" for value in values}

    def get_criterias(self):
        return self.__splitting_criterias

    def to_string(self):
        if (self.__is_leaf):
            return "{c} : {l}".format(c=self.__classification_attr, l=self.__classification_label)
        return "{}?".format(self.__node_attribute)

    def is_leaf(self):
        return self.__is_leaf

    def add_child(self, condition, child):
        self.__splitting_criterias[condition] = child
        self.__is_leaf = False

    def traverse_decision_tree(self, condition):
        if self.__is_leaf:
            return False
        try:
            x = self.__children[self.__splitting_criterias.index()]
            return x
        except:
            pass
        finally:
            return
