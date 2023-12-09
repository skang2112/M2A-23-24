import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(mse, (yhat, y)))

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)
X = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5, 4, requires_grad=True, dtype=torch.float64)
b = torch.randn(4, requires_grad=True, dtype=torch.float64)
print("LINEAR CHECK " , torch.autograd.gradcheck(linear, (X, W, b)))