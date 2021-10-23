import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# import helper

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

# create Random Forest Classifier and do 5-fold CV
print("5-fold cross-validation...")
xs = range(1, 750, 10)
scores = []
max_score = -1
best_x = 0

for x in xs:
    print(x)
    clf = RandomForestClassifier(n_estimators=x)
    cv_acc = cross_val_score(clf, Xs_train, ys_train, cv=5, n_jobs=-1)
    scores.append(cv_acc.mean())
    if cv_acc.mean() > max_score: 
        max_score = cv_acc.mean()
        best_x = x

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xs, scores, color = 'tab:orange', label = "training accuracy")
ax.legend()
ax.set_xlabel("# of estimators")
ax.set_ylabel("Accuracy")

plt.savefig("./reports/img/rf_acc.png")

# train on full training dataset
print("training final version...")
clf = RandomForestClassifier(n_estimators=best_x)
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

out.to_csv("./submissions/rf_submit.csv", index=False)