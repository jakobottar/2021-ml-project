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