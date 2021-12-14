import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

def one_hot(x):
    num_classes = len(np.unique(x))
    targets = x[np.newaxis].reshape(-1)
    one_hot_targets = np.eye(num_classes)[targets]
    return one_hot_targets.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, datafile, label_col):
        raw = pd.read_csv(datafile)

        s = (raw.dtypes == 'object')
        object_cols = list(s[s].index)

        le = LabelEncoder()
        data = raw.copy()

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
        self.ys = one_hot(np.array(ys, dtype=int))
    
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(torch.reshape(pred, y.shape), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss.append(loss.item())

    print(f"training error: {np.mean(train_loss):>8f}")
    return train_loss

def comparison(pred, target):
    pred = pred.round()
    print(pred.astype(int))
    print(target)
    print("\n")
    return np.sum(pred == target)

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(torch.reshape(pred, y.shape), y).item()
            accuracy += comparison(torch.reshape(pred, y.shape).cpu().numpy(), y.cpu().numpy())
    test_loss /= num_batches
    accuracy /= len(dataloader.dataset)
    print(f"test error: {test_loss:>8f} \n accuracy: {accuracy}\n")
    return test_loss

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_data = RegressionDataset('./data/train.csv', 'income>50K')
test_data = RegressionDataset('./data/test.csv', None) # TODO: handle test data's lack of label col

# Create data loaders.
batch_size = 10
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in train_dataloader:
    print("x:", x)
    print("x.shape:", x.shape)
    print("y:", y)
    print("y.shape:", y.shape)
    break

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

# widths = [14, 25, 50, 100, 150]
widths = [25]
# depths = [5, 9, 13]
depths = [9]

for width in widths:
    for depth in depths:

        print(f"{depth}-deep, {width}-wide network:\n-------------------------------")
        # Define model
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super(NeuralNetwork, self).__init__()
                self.input = nn.Sequential(nn.Linear(14, width), nn.LeakyReLU())
                self.body = nn.ModuleList([])
                for i in range(depth-2):
                    self.body.append(nn.Sequential(nn.Linear(width, width), nn.LeakyReLU()))
                self.out = nn.Sequential(nn.Linear(width, 2), nn.Sigmoid())

            def forward(self, x):
                x = self.input(x)
                for layer in self.body:
                    x = layer(x)
                res = self.out(x)
                return res

        model = NeuralNetwork().to(device)
        model.apply(init_xavier)

        loss_fn = nn.MSELoss()
        # loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        train_losses = np.array([])
        epochs = 100
        for t in range(epochs):
            print(f"epoch {t+1}", end=' ')
            epoch_losses = train(train_dataloader, model, loss_fn, optimizer)
            train_losses = np.append(train_losses, epoch_losses)

        fig, ax = plt.subplots()
        ax.plot(train_losses)
        ax.set_title(f"PyTorch: {depth}-deep, {width}-wide network")
        ax.set_xlabel("iteration")
        ax.set_ylabel("squared error")
        plt.savefig(f"./out/torch_d{depth}_w{width}.png")
        plt.close()
        
        test(train_dataloader, model, loss_fn) # all zeroes: 0.75936

print("Done!\nPlots saved in './out/'")