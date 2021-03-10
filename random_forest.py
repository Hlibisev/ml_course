from tree import *
import numpy as np
from scipy.stats import mode


class RandomForestClassifier:
    def __init__(self, criterion: str = "gini", max_depth: int = None, min_samples_leaf: int = 10,
                 max_iter: int = np.inf, n_estimators: int = 10, feat_bag: bool = True,
                 freq_feat_bag: float = None, out_of_bag: bool = True):
        """
        :param criterion: criterion of separation. "gini" or "entopy"
        :param max_depth: max depth of trees
        :param min_samples_leaf: how many samples should be at leaf to avoid separation
        :param max_iter: max iterations at finding best split values
        :param n_estimators: number of trees
        :param feat_bag: use feature bagging?
        :param freq_feat_bag: what part of feature is used at separation. Basically sqrt(num_feat)
        :param out_of_bag: calculate out of bag error?
        """

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.feat_bag = feat_bag
        self.freq_feat_bag = freq_feat_bag
        self.out_of_bag = out_of_bag
        self.OOB_error = 0
        self.out_of_indexes = []

    def fit(self, X, y):
        self.n_class = len(np.unique(y))
        self.trees = np.empty(self.n_estimators, dtype=object)
        for i in range(self.n_estimators):
            self.trees[i] = DecisionTreeClassifier(self.criterion, self.max_depth,
                                                   self.min_samples_leaf, self.max_iter,
                                                   self.feat_bag, self.freq_feat_bag)

            bag_ind = np.random.choice(len(X), len(X))
            self.trees[i].fit(X[bag_ind], y[bag_ind])

            # out of bag
            if self.out_of_bag:
                out_ind = np.setdiff1d(np.arange(len(X)), bag_ind)
                self.out_of_indexes.append(out_ind)

        if self.out_of_bag:
            self.OOB_error = self.compute_oob(X, y)

    def compute_oob(self, X, y, metric=None):
        out_predicts = [[] for _ in range(len(X))]

        # get out prediction
        for i, tree in enumerate(self.trees):
            out_ind = self.out_of_indexes[i]
            out_predict = tree.predict(X[out_ind])
            for i, ind in enumerate(out_ind):
                out_predicts[ind].append(out_predict[i])

        # calculate oob
        count, OOB_error = 0, 0
        for i, predict in enumerate(out_predicts):
            if predict:
                if metric is None:
                    OOB_error += (mode(predict)[0][0] == y[i])
                else:
                    OOB_error += metric(predict, y[i])
                count += 1
        return OOB_error / count

    def predict_proba(self, X, method="hard"):
        assert method in ("hard", "soft"), "method sould be hard or soft"

        pred = np.zeros((len(X), self.n_class))
        for tree in self.trees:
            if method == "hard":
                ind_c = np.array(tree.predict(X)).reshape(-1, 1)
                values = np.take_along_axis(pred, ind_c, axis=1) + 1
                np.put_along_axis(pred, ind_c, values, axis=1)
            else:
                raise NotImplementedError
        return pred / len(self.trees)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
