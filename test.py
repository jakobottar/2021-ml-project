import numpy as np

labs = np.array([0., 1., 0.001, 0.49, 0.51, 0.999])

print(labs)
print(labs.round())
labs = labs.round()
print(labs.astype(int))

# print()

# def one_hot(x):
#     num_classes = len(np.unique(x))
#     targets = x[np.newaxis].reshape(-1)
#     one_hot_targets = np.eye(num_classes)[targets]
#     return one_hot_targets

# print(one_hot(labs))