import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import helper

print("loading training data...")
raw_train = pd.read_csv("./data/train.csv")
print(raw_train)


# age - Continuous #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(raw_train['age'], bins = 20, color = '#fb293a')
ax.set_ylabel('Count')
ax.set_xlabel('Age (years)')
plt.savefig("./reports/img/age.png")
# plt.show()
plt.close()

# workclass - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['workclass'].value_counts().index
counts = raw_train['workclass'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.325)
ax.set_ylabel('Count')
ax.set_xlabel('workclass')
plt.savefig("./reports/img/workclass.png")
# plt.show()
plt.close()

# fnlwgt - Continuous #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(raw_train['fnlwgt'], bins=20, color = '#fb293a')
ax.set_ylabel('Count')
ax.set_xlabel('fnlwgt')
plt.savefig("./reports/img/fnlwgt.png")
# plt.show()
plt.close()

# education - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['education'].value_counts().index
counts = raw_train['education'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.31)
ax.set_ylabel('Count')
ax.set_xlabel('Education')
plt.savefig("./reports/img/education.png")
# plt.show()
plt.close()

# education-num - Continuous #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(raw_train['education.num'], bins=20, color = '#fb293a')
ax.set_ylabel('Count')
ax.set_xlabel('Education (years)')
plt.savefig("./reports/img/education-num.png")
# plt.show()
plt.close()

# marital-status - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['marital.status'].value_counts().index
counts = raw_train['marital.status'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.45)
ax.set_ylabel('Count')
ax.set_xlabel('Marital Status')
plt.savefig("./reports/img/marital-status.png")
# plt.show()
plt.close()

# occupation - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['occupation'].value_counts().index
counts = raw_train['occupation'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.38)
ax.set_ylabel('Count')
ax.set_xlabel('Occupation')
plt.savefig("./reports/img/occupation.png")
# plt.show()
plt.close()

# relationship - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['relationship'].value_counts().index
counts = raw_train['relationship'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.325)
ax.set_ylabel('Count')
ax.set_xlabel('Relationship')
plt.savefig("./reports/img/relationship.png")
# plt.show()
plt.close()

# race - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['race'].value_counts().index
counts = raw_train['race'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.36)
ax.set_ylabel('Count')
ax.set_xlabel('Race')
plt.savefig("./reports/img/race.png")
# plt.show()
plt.close()

# sex - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['sex'].value_counts().index
counts = raw_train['sex'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.2)
ax.set_ylabel('Count')
ax.set_xlabel('Sex')
plt.savefig("./reports/img/sex.png")
# plt.show()
plt.close()

# capital-gain - Continuous #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(raw_train['capital.gain'], bins=20, color = '#fb293a')
ax.set_ylabel('Count')
ax.set_xlabel('Capital Gain')
plt.yscale('log')
plt.savefig("./reports/img/capital-gain.png")
# plt.show()
plt.close()

# capital-loss - Continuous #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(raw_train['capital.loss'], bins=20, color = '#fb293a')
ax.set_ylabel('Count')
ax.set_xlabel('Capital Loss')
plt.yscale('log')
plt.savefig("./reports/img/capital-loss.png")
# plt.show()
plt.close()

# hours-per-week - Continuous #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(raw_train['hours.per.week'], bins=20, color = '#fb293a')
ax.set_ylabel('Count')
ax.set_xlabel('Work hours per week')
plt.savefig("./reports/img/hours-per-week.png")
# plt.show()
plt.close()

# native-country - Categorical #################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['native.country'].value_counts().index
counts = raw_train['native.country'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.5)
ax.set_ylabel('Count')
ax.set_xlabel('Native Country')
plt.yscale('log')
plt.savefig("./reports/img/native-country.png")
# plt.show()
plt.close()

# income>50k - Binary #################################

print(raw_train['income>50K'].value_counts())

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
values = raw_train['income>50K'].value_counts().index
counts = raw_train['income>50K'].value_counts().values
ax.bar(values, counts, color = '#fb293a')
ax.set_ylabel('Count')
ax.set_xlabel('Label: income>50K')
plt.savefig("./reports/img/income>50K.png")
# plt.show()
plt.close()

# Results Plot #################################

res = pd.read_csv("./submissions/results.txt")
print(res)

x = np.arange(len(res['method'].unique()))
fig, ax = plt.subplots(figsize=(8, 6))

# Define bar width. We'll use this to offset the second bar.
bar_width = 0.4

# Note we add the `width` parameter now which sets the width of each bar.
b1 = ax.bar(x, res['training_acc'], width=bar_width, label = "training accuracy")
b2 = ax.bar(x+bar_width, res['testing_acc'], width=bar_width, label = "testing accuracy")
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(res['method'].unique())
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.25)
ax.legend()
ax.set_ylabel('Accuracy')
ax.set_xlabel('Model')

plt.savefig("./reports/img/results.png")
# plt.show()
plt.close()