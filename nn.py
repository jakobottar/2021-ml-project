import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

batch_size = 10

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class RegressionDataset(Dataset):
    def __init__(self, datafile, label_col, drop_col = None, encoder = "ohe"):
        raw = pd.read_csv(datafile)

        if drop_col:
            for col in drop_col:
                raw.drop(col, axis=1, inplace=True)

        if encoder == 'ohe':
            obj_only = raw.select_dtypes('object')

            s = (raw.dtypes == 'object')
            object_cols = list(s[s].index)

            ohe = OneHotEncoder(handle_unknown='ignore', drop='if_binary')
            ohe.fit(obj_only)

            encoded = ohe.transform(obj_only).toarray()
            feature_names = ohe.get_feature_names_out(object_cols)

            data = pd.concat([raw.select_dtypes(exclude='object'), 
                    pd.DataFrame(encoded,columns=feature_names).astype(int)], axis=1)
        
        if encoder == 'le':
            le = LabelEncoder()
            data = raw.copy()

            s = (raw.dtypes == 'object')
            object_cols = list(s[s].index)

            # LabelEncode the categorical variables
            for col in object_cols:
                data[col+'num'] = le.fit_transform(data[col])
                data.drop(col, axis=1, inplace=True)

        # create Features/Labels dfs
        if label_col == None:
            xs = data
            ys = np.zeros((len(data),))
        else:
            xs = data.drop(label_col, axis=1)
            ys = data[label_col]

        self.xs = np.array(xs, dtype=np.float32)
        self.ys = np.array(ys, dtype=int)
    
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            # print("pred: ", pred.argmax(1).cpu().numpy())
            # print("targ: ", y.cpu().numpy())
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

def predict(dataloader, model):
    model.eval()
    res = np.array([])
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)
            res = np.append(res, pred.argmax(1).cpu().numpy())
    return res
# Define model
DEPTH = 13
WIDTH = 50
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Sequential(nn.Linear(14, WIDTH), nn.LeakyReLU())
        self.body = nn.ModuleList([])
        for i in range(DEPTH-2):
            self.body.append(nn.Linear(WIDTH, WIDTH))
            self.body.append(nn.LeakyReLU())
        self.out = nn.Linear(WIDTH, 2)

    def forward(self, x):
        x = self.input(x)
        for layer in self.body:
            x = layer(x)
        res = self.out(x)
        return res

# Create datasets
train_data = RegressionDataset('./data/train.csv', 'income>50K', encoder='le')
test_data = RegressionDataset('./data/test.csv', None, drop_col=['ID'], encoder='le')

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in train_dataloader:
    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

model = NeuralNetwork().to(device)
model.apply(init_xavier)
print(model)

# perc 0 in trianing data: 0.75936
weights = [1-0.75936, 0.75936]
class_weights = torch.FloatTensor(weights).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

best_model = model
best_acc = 0

epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    acc = test(train_dataloader, model, loss_fn)
    if acc > best_acc:
        best_acc = acc
        best_model = model
print(f"Our best training accuracy was {(100*best_acc):>0.1f}%")
filename = "./out/model.pth"
torch.save(model, filename)
print(f"saved model to {filename}")

# model = torch.load(filename).to(device)
pred_test = predict(test_dataloader, best_model)

data = {'ID': list(range(1, len(pred_test)+1)),
        'Prediction': pred_test}
out = pd.DataFrame(data)

out.to_csv("./submissions/nn_submit.csv", index=False)
print("Done!")