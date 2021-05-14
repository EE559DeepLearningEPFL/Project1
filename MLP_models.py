# MLP family: MLP, MLP + weight sharing, MLP + auxiliary loss, MLP + weight sharing + auxiliary loss
from torch import nn
from torch.nn import functional as F
import torch

# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(392, 160)
        self.fc2 = nn.Linear(160, 64) 
        self.fc3 = nn.Linear(64, 2) 


    def forward(self, x):
        x = x.view(-1,392) # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
        
# MLP + weight sharing
class SiameseMLP(nn.Module):
    def __init__(self):
        super(SiameseMLP, self).__init__()
        
        self.fc1 = nn.Linear(196, 160)
        self.fc2 = nn.Linear(160, 10)
        self.fc3 = nn.Linear(20, 2) 


    def forward(self, x1, x2):
        x1 = x1.view(-1,196) # flatten
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        x1 = F.relu(x1)
        
        x2 = x2.view(-1,196) # flatten
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        x2 = F.relu(x2)
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.sigmoid(self.fc3(x))
        
        return x

# MLP + auxiliary loss
class AuxMLP(nn.Module):
    def __init__(self):
        super(AuxMLP, self).__init__()
        
        self.fc11 = nn.Linear(196, 160)
        self.fc12 = nn.Linear(196, 160)
        self.fc21 = nn.Linear(160, 10)
        self.fc22 = nn.Linear(160, 10)
        self.fc3 = nn.Linear(20, 2) 


    def forward(self, x1, x2):
        x1 = x1.view(-1,196) # flatten
        x1 = F.relu(self.fc11(x1))
        x1 = self.fc21(x1)
        aux1 = F.softmax(x1)
        x1 = F.relu(x1)
        
        x2 = x2.view(-1,196) # flatten
        x2 = F.relu(self.fc12(x2))
        x2 = self.fc22(x2)
        aux2 = F.softmax(x2)
        x2 = F.relu(x2)
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.sigmoid(self.fc3(x))
        
        return x, aux1, aux2

# MLP + weight sharing + auxiliary loss
class AuxsiameseMLP(nn.Module):
    def __init__(self):
        super(AuxsiameseMLP, self).__init__()
        
        self.fc1 = nn.Linear(196, 160)
        self.fc2 = nn.Linear(160, 10)
        self.fc3 = nn.Linear(20, 2) 


    def forward(self, x1, x2):
        x1 = x1.view(-1,196) # flatten
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        aux1 = F.softmax(x1)
        x1 = F.relu(x1)
        
        x2 = x2.view(-1,196) # flatten
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        aux2 = F.softmax(x2)
        x2 = F.relu(x2)
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.sigmoid(self.fc3(x))
        
        return x, aux1, aux2









