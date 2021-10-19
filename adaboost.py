import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import helper

data, header = helper.LoadDataset("./data/train.csv")

Xs = np.array([d[:-1] for d in data])
ys = np.array([d[-1] for d in data])

clf = AdaBoostClassifier(n_estimators=100)
# scores = cross_val_score(clf, Xs, ys, cv=5)
# print(scores.mean())