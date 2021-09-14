import torch
from torch import nn, optim

class ANN(nn.Module):
  def __init__(self):
    super().__init__()
    self.input = nn.Linear(784, 100)
    self.bn1 = nn.BatchNorm1d(100)
    self.hidden = nn.Linear(100, 50)
    self.bn2 = nn.BatchNorm1d(50)
    self.output = nn.Linear(50, 10)
    self.bn3 = nn.BatchNorm1d(10)

  def forward(self, x):
    y = self.bn1(self.input(x))
    y = self.bn2(self.hidden(y))
    y = self.bn3(self.output(y))
    return y
