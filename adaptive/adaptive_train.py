import numpy as np
import os
from configs import *
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# load data
data_set = '/mnt/local/Baselines_Bugs/DrQA/data/adaptive_data/train.txt'
X = []
y = []
with open(data_set, 'r') as f:
    for line in f:
        line = line.strip().split(',')
        y.append(int(line[0]))
        X.append(np.fromstring(line[1], dtype=float, sep=','))

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=3407)

# convert data to tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train).view(-1,1)
X_dev_tensor = torch.Tensor(X_dev)
y_dev_tensor = torch.Tensor(y_dev).view(-1,1)

# Create data loaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
dev_data = TensorDataset(X_dev_tensor, y_dev_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=32, shuffle=True)

# Define the model (simple linear regression)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# # setup device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create model
input_dim = X_train_tensor.shape[1]
model = LinearRegressionModel(input_dim).to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # move data to device
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate on dev set
model.eval()
with torch.no_grad():
    predictions = []
    for inputs, labels in dev_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # move data to device
        outputs = model(inputs)
        rounded_preds = torch.round(outputs)  # make the predictions ordinal
        predictions.extend(rounded_preds.cpu().numpy().flatten().tolist())  # move data back to CPU for further processing

# count occurrences of each class
b = 1
print(np.bincount(np.array(predictions).astype(np.int32) + b))

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_dev_tensor.cpu().numpy(), np.array(predictions))
print(f'Mean Squared Error: {mse}')


def get_topk_recall_and_mrr(y_true, y_pred_proba, k):
    # Get the top k predictions for each example
    top_k_preds = np.argsort(-y_pred_proba, axis=1)[:, :k]
    
    # Compute Top-k recall
    recall = np.any(top_k_preds.T == y_true, axis=0).mean()
    
    # Compute MRR
    first_correct_indices = np.argmax(top_k_preds.T == y_true, axis=0)
    mrr = (1 / (first_correct_indices + 1)).mean()

    return recall, mrr

# You can call it like this
recall, mrr = get_topk_recall_and_mrr(y_dev, y_pred_proba, k=5)
ab = np.argpartition()


# save the model
torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
