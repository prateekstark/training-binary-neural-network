import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.datasets
from Synthetic_Models.twomoons_model import TwoMoonsMLP, TwoMoonsSTE
from BayesBiNN_Optimizer import BiNNOptimizer as BayesBiNN
from tqdm import tqdm


num_points_train=200
num_points_test = 200
X_orig, y_train = sklearn.datasets.make_moons(num_points_train, noise=0.1, random_state=0)
X_test, y_test = sklearn.datasets.make_moons(num_points_test, noise=0.1, random_state=0)
X_orig = torch.from_numpy(X_orig).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(1)
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().unsqueeze(1)
X_mean = X_orig.mean(dim=0)
X_std = X_orig.std(dim=0)
X_train = (X_orig-X_mean)/X_std
X_test = (X_test-X_mean)/X_std




# Training setting for BayesBiNN

learning_rate = 1e-3
learning_rate_decay = 0.1
learning_rate_decay_epochs = [1500, 2500]
beta = 0.99
temperature = 1
initialize_lambda = 15
mc_test_samples = 10
N = 5
hidden_layers = [64,64]
epochs = 3000
criterion = nn.BCEWithLogitsLoss()


# Training BayesBiNN Network

torch.manual_seed(0)
np.random.seed(0)

model_binn = TwoMoonsMLP(X_train.shape[1], hidden_layers, y_train.shape[1])

optimizer_binn = BayesBiNN(model_binn, train_set_size=X_train.shape[0], lr=learning_rate, beta=beta, 
                 N=N, temperature=temperature, initialize_lambda=initialize_lambda)

for epoch in tqdm(range(epochs)):
    if epoch in learning_rate_decay_epochs:
        for params in optimizer_binn.param_groups:
            params['lr'] *=  learning_rate_decay
    def closure():
        optimizer_binn.zero_grad()
        y_pred = model_binn.forward(X_train)
        loss = criterion(y_pred, y_train)
        return loss, y_pred
    loss, y_pred = optimizer_binn.step(closure)


# Training setting for STE Network 

learning_rate = 1e-1
learning_rate_decay = 0.1
learning_rate_decay_epochs = [1500, 2500]
hidden_layers = [64,64]
epochs = 3000
criterion = nn.BCEWithLogitsLoss()


# Training STE Network 

torch.manual_seed(0)
np.random.seed(0)

model_ste = TwoMoonsSTE(X_train.shape[1], hidden_layers, y_train.shape[1])
optimizer_ste = optim.Adam(model_ste.parameters(), lr=learning_rate)

for i in tqdm(range(epochs)):
    if i in learning_rate_decay_epochs:
        for param_group in optimizer_ste.param_groups:
            param_group['lr'] *=  learning_rate_decay
    
    optimizer_ste.zero_grad()
    y_pred = model_ste.forward(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    
    for p in list(model_ste.parameters()):
        if hasattr(p, 'org'):
            p.data.copy_(p.org)

    optimizer_ste.step()

    for p in list(model_ste.parameters()):
        if hasattr(p, 'org'):
            p.org.copy_(p.data.clamp_(-1, 1))



def setup_grid(X, h=0.01, padding=0.5):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    return grid, xx, yy

# Plotting decision boundary for Two Moons Dataset

yf = y_train.flatten()
grid_orig, xx, yy = setup_grid(X_orig, h=0.01, padding=0.3)
grid = (grid_orig - X_mean) / X_std

fig, axes = plt.subplots(3,1,figsize=(7,18))

loc_y = plticker.MultipleLocator(base=0.5)
loc_x = plticker.MultipleLocator(base=1.0)

axes[0].tick_params(axis='both', which='major', labelsize=16)
axes[0].tick_params(axis='both', which='minor', labelsize=16)
axes[1].tick_params(axis='both', which='major', labelsize=16)
axes[1].tick_params(axis='both', which='minor', labelsize=16)
axes[2].tick_params(axis='both', which='major', labelsize=16)
axes[2].tick_params(axis='both', which='minor', labelsize=16)

ms = 100

Z = model_ste.predict(grid).reshape(xx.shape)
c = axes[0].contourf(xx, yy, Z, alpha=0.7, antialiased=True, cmap='PuOr')
axes[0].scatter(X_orig[yf==0, 0], X_orig[yf==0, 1], s=ms, lw=1.5, edgecolors='black', color='#EFBC38', marker='o')
axes[0].scatter(X_orig[yf==1, 0], X_orig[yf==1, 1], s=ms, lw=1.5, edgecolors='black',color='#CE84EE',marker = 'p')
axes[0].xaxis.set_major_locator(loc_x)
axes[0].yaxis.set_major_locator(loc_y)
axes[0].set_ylabel('STE Prediction', fontsize=18)

Z = optimizer_binn.montecarlo_predictions(model_binn.forward,grid,epsilons=None)
Z = torch.sigmoid(torch.stack(Z)).mean(dim=0).reshape(xx.shape).detach()
c = axes[1].contourf(xx, yy, Z, alpha=0.7, antialiased=True, cmap='PuOr')
axes[1].scatter(X_orig[yf==0, 0], X_orig[yf==0, 1], s=ms, edgecolors='black', color='#EFBC38',lw=1.5, marker = 'o')
axes[1].scatter(X_orig[yf==1, 0], X_orig[yf==1, 1], s=ms, edgecolors='black', color='#CE84EE',lw=1.5, marker = 'p')
axes[1].xaxis.set_major_locator(loc_x)
axes[1].yaxis.set_major_locator(loc_y)
axes[1].set_ylabel('BayesBiNN Mode', fontsize=18)

torch.manual_seed(0)
n_samples = 10
raw_noises = []
for mc_sample in range(n_samples):
    raw_noises.append(torch.bernoulli(torch.sigmoid(2*optimizer_binn.state['lambda'])))
Z = optimizer_binn.montecarlo_predictions(model_binn.forward, grid, epsilons=raw_noises)
Z = torch.sigmoid(torch.stack(Z)).mean(dim=0).reshape(xx.shape).detach()
c = axes[2].contourf(xx, yy, Z, alpha=0.7, antialiased=True, cmap='PuOr')
axes[2].scatter(X_orig[yf==0, 0], X_orig[yf==0, 1], s=ms, edgecolors='black', color='#EFBC38',lw=1.5, marker = 'o')
axes[2].scatter(X_orig[yf==1, 0], X_orig[yf==1, 1], s=ms, edgecolors='black', color='#CE84EE',lw=1.5, marker = 'p')
axes[2].xaxis.set_major_locator(loc_x)
axes[2].yaxis.set_major_locator(loc_y)
axes[2].set_ylabel('BayesBiNN Mean', fontsize=18)

plt.show()