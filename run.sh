#!/bin/bash
echo "running AdaBoost"
python adaboost.py

echo "running RandomForest"
python rf.py

echo "running SVM"
python svm.py

echo "running Naive Bayes"
python bayes.py

echo "running Logistic Regression"
python lr.py

echo "running Neural Net"
python nn.py

# kaggle competitions submit -c ilp2021f -f submission.csv -m "Message"