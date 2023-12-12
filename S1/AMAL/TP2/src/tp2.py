import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
import torch.nn as nn


writer = SummaryWriter()

### LOADING DATA ### 
data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

### FOR TRAIN/TEST SPLITTING ###
train_size = round(datax.shape[0]*0.8)
idx = torch.randperm(datax.shape[0])
train_idx, test_idx = idx[:train_size], idx[train_size:]

train_x, train_y = datax[train_idx], datay[train_idx]
test_x, test_y = datax[test_idx], datay[test_idx]

### DEFINE MODEL ###
eps = 10e-4
EPOCHS = 1000
latent_size = 8
batch_size = 64 #for batch training, take batch_size = datax.shape[0]

model = nn.Sequential(
    nn.Linear(datax.shape[1], latent_size),
    nn.Tanh(),
    nn.Linear(latent_size, datay.shape[1])
)

MSE = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=eps)

### TRAIN ###

with torch.no_grad():
    print("Start loss :", MSE(model(train_x), train_y).item())

for n_iter in tqdm(range(EPOCHS)):
    batch_idx = torch.randint(0, train_x.shape[0], (batch_size,))

    x, y = train_x[batch_idx], train_y[batch_idx]
    yhat = model(x)
    loss = MSE(yhat, y)

    writer.add_scalar('Loss/train', loss.item(), n_iter)
    with torch.no_grad():
        writer.add_scalar('Loss/test', MSE(model(test_x), test_y).item(), n_iter)

    loss.backward()
    
    optimiser.step()
    optimiser.zero_grad()
  
with torch.no_grad():
    print("End loss :", MSE(model(train_x), train_y).item())
    print("End test loss :", MSE(model(test_x), test_y).item())