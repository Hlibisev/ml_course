import numpy as np
from typing import Set


def get_freq(x):
    """ Calculating frequencies unique values at x """
    count = np.bincount(x)
    ind = np.nonzero(count)
    return count[ind] / len(x)


def gini(x=None, freq=None):
    """ Calculating gini criterion """
    if freq is None and x is not None:
        freq = get_freq(x)
    return sum(freq * (1 - freq))


def entropy(x=None, freq=None):
    """ Calculating entropy criterion """
    if freq is None and x is not None:
        freq = get_freq(x)
    return - sum(freq * np.log2(freq))


def gain(criterion=gini, left_y=None, right_y=None):
    """ Calculating gain """
    return -1 * (len(left_y) * criterion(left_y) + len(right_y) * criterion(right_y))


# quick analog
def gain2(criterion, right_sum, left_sum, right_freq, left_freq):
    """ Calculating gain2 """
    return -1 * (left_sum * criterion(freq=left_freq) + right_sum * criterion(freq=right_freq))


class DecisionTreeNode:
    def __init__(self, depth=0, ind=None):
        self.split_dim = None
        self.split_value = None
        self.left = None
        self.right = None
        self.ind = ind if ind is not None else []
        self.size = len(self.ind)
        self.depth = depth

    def define(self, split_dim, split_value, left, right):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


class DecisionTreeClassifier:
    def __init__(self, criterion: str = "gini", max_depth: int = 20, min_samples_leaf: int = 1,
                 max_iter: int = np.inf, feat_bag: bool = False, freq_feat_bag: float = None):
        """
        :param criterion: criterion of separation. "gini" or "entopy"
        :param max_depth: max depth of tree
        :param min_samples_leaf: how many samples should be at leaf to avoid separation
        :param max_iter: max iterations at finding best split values
        :param feat_bag: use feature bagging?
        :param freq_feat_bag: what part of feature is used at separation. Basically sqrt(num_feat)
        """
        self.root = None
        self.criterion = gini
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.criterion = gini if criterion == "gini" else entropy
        self.max_iter = max_iter
        self.feat_bag = feat_bag
        self.freq_feat_bag = freq_feat_bag
        self.number_feat_bag = None

    def fit(self, X, y):
        if self.freq_feat_bag is None:
            self.number_feat_bag = int(np.sqrt(X.shape[1]))
        else:
            self.number_feat_bag = int(X.shape[1] * self.freq_feat_bag)

        self.y = y
        self.number_of_classes = len(np.unique(y))
        self.root = DecisionTreeNode(ind=np.arange(len(X)))
        nodes = {self.root}  # type: Set[DecisionTreeNode]

        while nodes:
            new_nodes = set()
            for node in nodes:
                if node.size > 2 * self.min_samples_leaf and node.depth < self.max_depth:
                    split_dim, split_value = self.find_split(X[node.ind], y[node.ind])

                    # if all y at same class
                    if split_dim is None: continue

                    sep = X[node.ind, split_dim] < split_value
                    left_ind, right_ind = node.ind[sep], node.ind[~sep]

                    # if all y at one side after separation
                    if len(left_ind) == 0 or len(right_ind) == 0:
                        continue

                    left, right = self.make_nodes(split_dim, split_value,
                                                  left_ind, right_ind, node)

                    new_nodes.add(left)
                    new_nodes.add(right)
            nodes = new_nodes

    @staticmethod
    def make_nodes(split_dim, split_value, left_ind, right_ind, node):
        left = DecisionTreeNode(ind=left_ind, depth=node.depth + 1)
        right = DecisionTreeNode(ind=right_ind, depth=node.depth + 1)
        node.define(split_dim, split_value, left, right)
        return left, right

    def predict_proba(self, X):
        ind = np.arange(len(X))
        return self._traversal(X, ind, self.root)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in probabilities]

    def find_split(self, X, y):
        quality = -np.inf
        split_dim, split_value = -1, -1  # feature position, impurity
        feature_and_y = np.vstack([X[:, 0], y]).T  # some feature, y
        unique_y = np.unique(y, return_counts=True)

        # If all y are same, do nothing
        if len(unique_y[0]) == 1:
            return None, None

        left_counter = np.zeros(self.number_of_classes)
        right_counter = np.zeros(self.number_of_classes)
        right_counter[unique_y[0]] += unique_y[1]

        # feature bagging
        if self.feat_bag is True:
            features = np.random.choice(X.shape[1], self.number_feat_bag, replace=False)
        else:
            features = [i for i in range(X.shape[1])]

        for feature in features:
            feature_and_y[:, 0] = X[:, feature]
            sorted_feat_and_y = np.array(sorted(feature_and_y, key=lambda x: x[0]))  # sort by feature

            values_X, indexes_of_unique = np.unique(sorted_feat_and_y[:, 0], return_index=True)

            # max iterations is equal max_iters
            if len(values_X) > self.max_iter:
                indexes = sorted(np.random.choice(len(values_X), self.max_iter, replace=False))
            else:
                indexes = indexes_of_unique

            # clear counters
            left_counter *= 0
            right_counter *= 0
            right_counter[unique_y[0]] += unique_y[1]
            left_sum = 0
            right_sum = len(y)

            for i in range(0, len(indexes)):  # examine y
                if i == 0 and indexes[i - 1] != 0:
                    ind1, ind2 = 0, indexes[i]
                else:
                    ind1, ind2 = indexes[i - 1], indexes[i]
                values_in_segment, counts = np.unique(sorted_feat_and_y[ind1: ind2, 1], return_counts=True)

                # update info about left and right
                left_counter[values_in_segment.astype(int)] += counts
                right_counter[values_in_segment.astype(int)] -= counts
                left_sum += sum(counts)
                right_sum -= sum(counts)

                # to evade warning
                if left_sum == 0: continue

                freq_left = left_counter / left_sum
                freq_right = right_counter / right_sum
                l, r = np.nonzero(freq_left), np.nonzero(freq_right)

                # calculate gain
                g = gain2(self.criterion, right_sum, left_sum, freq_right[r], freq_left[l])

                if g > quality:
                    split_dim, split_value = feature, sorted_feat_and_y[ind2, 0]
                    quality = g
        return split_dim, split_value

    def _traversal(self, X, ind, node):
        if node.is_leaf():
            values, counts = np.unique(self.y[node.ind], return_counts=True)
            freq = counts / sum(counts)
            predict = dict(zip(values, freq))
            return [predict for _ in range(len(ind))]

        separation = X[ind, node.split_dim] < node.split_value
        left_ind, right_ind = ind[separation], ind[~separation]

        predict = np.zeros(len(ind), dtype=object)
        predict[separation] = self._traversal(X, left_ind, node.left)
        predict[~separation] = self._traversal(X, right_ind, node.right)
        return predict
