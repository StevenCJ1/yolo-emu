import torch.nn as nn
import numpy as np
import torch
np.random.seed(42)


class MLP1(nn.Module):
    def __init__(self, in_chs) -> None:
        super(MLP1, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_chs, 512)
        self.l2 = nn.Linear(512, 256)
        
    def forward(self, x):
        x = x.reshape(-1, 3072)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return x
    
    
class MLP2(nn.Module):
    def __init__(self) -> None:
        super(MLP2, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 64)
        
    def forward(self, x):
        x = x.reshape(-1, 256)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return x    
    
    
class MLP3(nn.Module):
    def __init__(self) -> None:
        super(MLP3, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 64)
        x = self.relu(self.l1(x))
        x = self.l2(x)
        x = torch.argmax(x, dim=1)
        return x