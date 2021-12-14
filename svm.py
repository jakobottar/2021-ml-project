import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# LabelEncode the categorical variables
for col in object_cols:
    train[col+'.numconv'] = le.fit_transform(train[col])
    train.drop(col, axis=1, inplace=True)

# create Features/Labels dfs
Xs_train = train.drop('income>50K', axis=1)
ys_train = train['income>50K']

weights = {0 : 0.25,
           1 : 0.75}

# create AdaBoost Classifier and do 5-fold CV
print("5-fold cross-validation...")
svc = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))
print(svc)
cv_acc = cross_val_score(svc, Xs_train, ys_train, cv=5, n_jobs=-1)
print(cv_acc.mean())

print("training on full dataset...")
svc_final = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))
svc_final = svc_final.fit(Xs_train, ys_train)

### Make Submission from test.csv
print("loading testing data...")
raw_test = pd.read_csv("./data/test.csv")

test = raw_test.copy()

test.drop('ID', axis=1, inplace=True)

# LabelEncode the categorical variables
for col in object_cols:
    test[col+'.numconv'] = le.fit_transform(test[col])
    test.drop(col, axis=1, inplace=True)

#predict resulting values
print("predicting from testing data...")
pred_test = svc_final.predict(test)

data = {'ID': list(range(1, len(pred_test)+1)),
        'Prediction': pred_test}
out = pd.DataFrame(data)

out.to_csv("./submissions/svm_submit.csv", index=False)
print("Done!")