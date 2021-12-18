import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# import helper

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but AdaBoostClassifier was fitted with feature names")

print("loading training data...")
raw_train = pd.read_csv("./data/train.csv")

# got info on categorical data handling from 
# https://analyticsindiamag.com/complete-guide-to-handling-categorical-data-using-scikit-learn/

# determine categorical variables
s = (raw_train.dtypes == 'object')
object_cols = list(s[s].index)

le = LabelEncoder()
train = raw_train.copy()
train.drop('education.num', axis=1, inplace=True) # I drop this because I want to do the conversion myself

# LabelEncode the categorical variables
for col in object_cols:
    train[col+'num'] = le.fit_transform(train[col])
    train.drop(col, axis=1, inplace=True)

# create Features/Labels dfs
Xs_train = train.drop('income>50K', axis=1)
ys_train = train['income>50K']

# create AdaBoost Classifier and do 5-fold CV
print("5-fold cross-validation...")
xs = np.arange(0, 810, 10)
xs[0] = 1
scores = []
max_score = -1
best_x = 0

for x in xs:
    clf = AdaBoostClassifier(n_estimators=x)
    cv_acc = cross_val_score(clf, Xs_train, ys_train, cv=5, n_jobs=-1)
    scores.append(cv_acc.mean())
    print(f"{x}: {cv_acc.mean():>4f}")
    if cv_acc.mean() > max_score: 
        max_score = cv_acc.mean()
        best_x = x

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xs, scores, color = 'tab:blue', label = "training accuracy")
ax.legend()
ax.set_xlabel("# of estimators")
ax.set_ylabel("Accuracy")

plt.savefig("./reports/img/ab_acc.png")

# train on full training dataset
print(f"training final version with {best_x} classifiers...")
clf = AdaBoostClassifier(n_estimators=best_x)
clf = clf.fit(Xs_train, ys_train)

### Make Submission from test.csv
print("loading testing data...")
raw_test = pd.read_csv("./data/test.csv")

test = raw_test.copy()
test.drop(['education.num', 'ID'], axis=1, inplace=True)

# LabelEncode the categorical variables
for col in object_cols:
    test[col+'num'] = le.fit_transform(test[col])
    test.drop(col, axis=1, inplace=True)

#predict resulting values
print("predicting from testing data...")
pred_test = clf.predict(test)

data = {'ID': list(range(1, len(pred_test)+1)),
        'Prediction': pred_test}
out = pd.DataFrame(data)

out.to_csv("./submissions/adaboost_submit.csv", index=False)