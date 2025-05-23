import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None, depth=0):
        self.feature = None
        self.children = {}
        self.prediction = None

        if len(set(y)) == 1:
            self.prediction = y[0]
            return

        if max_depth is not None and depth >= max_depth:
            self.prediction = Counter(y).most_common(1)[0][0]
            return

        best_gain = 0
        best_feature = None
        best_splits = None

        for i in range(len(X[0])):
            values = set(row[i] for row in X)
            splits = {v: [] for v in values}
            y_splits = {v: [] for v in values}
            for row, label in zip(X, y):
                splits[row[i]].append(row)
                y_splits[row[i]].append(label)

            gain = self._information_gain(y, y_splits.values())
            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_splits = (splits, y_splits)

        if best_gain < threshold or best_feature is None:
            self.prediction = Counter(y).most_common(1)[0][0]
            return

        self.feature = best_feature
        for value in best_splits[0]:
            self.children[value] = DecisionTree(
                best_splits[0][value],
                best_splits[1][value],
                threshold=threshold,
                max_depth=max_depth,
                depth=depth + 1
            )

    def _entropy(self, labels):
        total = len(labels)
        counts = Counter(labels)
        return -sum((count / total) * np.log2(count / total) for count in counts.values() if count > 0)

    def _information_gain(self, parent, children):
        total = len(parent)
        parent_entropy = self._entropy(parent)
        weighted_entropy = sum((len(child) / total) * self._entropy(child) for child in children)
        return parent_entropy - weighted_entropy

    def _majority_class(self):
        if self.prediction is not None:
            return self.prediction
        labels = []
        for child in self.children.values():
            labels.append(child._majority_class())
        return Counter(labels).most_common(1)[0][0]

    def predict(self, x):  # Ex: x = ['apple', 'green', 'circle']
        if self.prediction is not None:
            return self.prediction
        value = x[self.feature]
        child = self.children.get(value)
        if child:
            return child.predict(x)
        else:
            return self._majority_class()

def train_decision_tree(X, y):
    return DecisionTree(X, y)