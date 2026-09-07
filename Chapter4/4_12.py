from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
X_with_bias = np.c_[np.ones(len(X)), X]
# print(X_with_bias)
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X_with_bias,y, test_size=0.2)

X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25)
y_train = pd.Series.to_numpy(y_train)
y_val = pd.Series.to_numpy(y_val)
y_test = pd.Series.to_numpy(y_test)

m = len(X_train)
n = len(X_val)

def get_yk(y):
    return np.diag(np.ones(y.max() + 1))[y]

Y_train_one_hot = get_yk(y_train)
Y_valid_one_hot = get_yk(y_val)
Y_test_one_hot = get_yk(y_test)

mean = X_train[:, 1:].mean(axis=0)
std = X_train[:, 1:].std(axis=0)
X_train[:, 1:] = (X_train[:, 1:] - mean) / std
X_val[:, 1:] = (X_val[:, 1:] - mean) / std
X_test[:, 1:] = (X_test[:, 1:] - mean) / std

def softmax(logits):
    exps = np.exp(logits)
    exp_sums = exps.sum(axis=1, keepdims=True)
    return exps / exp_sums

n_inputs = X_train.shape[1] 
n_outputs = len(np.unique(y_train))


eta = 0.5
n_epochs = 5001
m = len(X_train)
epsilon = 1e-5
C = 100
best_loss = np.inf

np.random.seed(42)
Theta = np.random.randn(n_inputs, n_outputs)

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    Y_proba_valid = softmax(X_val @ Theta)
    xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()
    total_loss = xentropy_losses.sum(axis=1).mean() + 1 / C * l2_loss
    if epoch % 1000 == 0:
        print(epoch, total_loss.round(4))
    if total_loss < best_loss:
        best_loss = total_loss
    else:
        print(epoch - 1, best_loss.round(4))
        print(epoch, total_loss.round(4), "early stopping!")
        break
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, n_outputs]), 1 / C * Theta[1:]]
    Theta = Theta - eta * gradients

logits = X_val @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)

accuracy_score = (y_predict == y_val).mean()
print(accuracy_score)

logits = X_test @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)

accuracy_score = (y_predict == y_test).mean()
print(accuracy_score)
