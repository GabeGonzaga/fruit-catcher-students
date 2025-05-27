import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None, depth=0):
        self.feature = None
        self.children = {}
        self.prediction = None

        # terminal conditions
        if len({*y}) == 1 or (max_depth is not None and depth >= max_depth):
            self.prediction = (y[0]
                if len({*y}) == 1
                else Counter(y).most_common(1)[0][0]
            )
            return

        # compute splits & gains for each feature
        def get_splits(i):
            vals = {row[i] for row in X}
            splits = {v: [row for row in X if row[i] == v] for v in vals}
            y_splits = {v: [lab for row, lab in zip(X, y) if row[i] == v] for v in vals}
            return splits, y_splits

        feats = (
            (i, splits, y_splits, self._information_gain(y, y_splits.values()))
            for i in range(len(X[0]))
            for splits, y_splits in [get_splits(i)]
        )
        best_i, best_splits, best_y_splits, best_gain = max(feats, key=lambda t: t[3], default=(None, None, None, 0))

        if best_gain < threshold or best_i is None:
            self.prediction = Counter(y).most_common(1)[0][0]
            return

        self.feature = best_i
        self.children = {
            v: DecisionTree(best_splits[v], best_y_splits[v], threshold, max_depth, depth + 1)
            for v in best_splits
        }

    def _entropy(self, labels):
        total = len(labels)
        counts = Counter(labels)
        return -sum((cnt/total) * np.log2(cnt/total)
                    for cnt in counts.values() if cnt)

    def _information_gain(self, parent, children):
        total = len(parent)
        p_ent = self._entropy(parent)
        c_ent = sum((len(c)/total) * self._entropy(c) for c in children)
        return p_ent - c_ent

    def _majority_class(self):
        if self.prediction is not None:
            return self.prediction
        return Counter(
            c._majority_class() for c in self.children.values()
        ).most_common(1)[0][0]

    def predict(self, x):
        return (
            self.prediction
            if self.prediction is not None else
            self.children[x[self.feature]].predict(x)
                if x[self.feature] in self.children else
            self._majority_class()
        )

def train_decision_tree(X, y):
    return DecisionTree(X, y)