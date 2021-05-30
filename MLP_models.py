# MLP family: MLP, MLP + weight sharing, MLP + auxiliary loss, MLP + weight sharing + auxiliary loss
from torch import nn
from torch.nn import functional as F
import torch

# MLP
class MLP(nn.Module):
    '''
    fc1, fc2
        two fully connected layers
    fc3
        output layer
    relu
        activation function for hidden layers
    sigmoid
        activation function for output layer
    '''
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(392, 160) # input size is n*2*14*14 = n*392, n is the number of samples
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
    '''
    basic structure similar to the MLP
    input is splited into two 1*14*14 images for separating training, share the same parameters
    '''
    def __init__(self):
        super(SiameseMLP, self).__init__()
        
        self.fc1 = nn.Linear(196, 160) # input size is n*1*14*14 = n*196
        self.fc2 = nn.Linear(160, 10)
        self.fc3 = nn.Linear(20, 2) 


    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = x1.view(-1,196) # flatten
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        x1 = F.relu(x1) # size n*10
        
        # train the second splitted image in second channel
        x2 = x2.view(-1,196) # flatten
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        x2 = F.relu(x2) # size n*10
        
        # concatenate the results of two channels
        x = torch.cat([x1, x2], dim=1) # size n*20
        
        x = torch.sigmoid(self.fc3(x)) # size n*2
        
        return x

# MLP + auxiliary loss
class AuxMLP(nn.Module):
    '''
    basic structure similar to the MLP
    input is splited into two 1*14*14 images for separating training, use different parameters
    softmax for the auxiliary output layers
    '''
    def __init__(self):
        super(AuxMLP, self).__init__()
        
        self.fc11 = nn.Linear(196, 160) # fc1 for first channel
        self.fc12 = nn.Linear(196, 160) # fc1 for second channel
        self.fc21 = nn.Linear(160, 10)  # fc2 for first channel
        self.fc22 = nn.Linear(160, 10)  # fc2 for second channel
        self.fc3 = nn.Linear(20, 2) 


    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = x1.view(-1,196) # flatten
        x1 = F.relu(self.fc11(x1))
        x1 = self.fc21(x1)
        
        # train the second splitted image in second channel
        x2 = x2.view(-1,196) # flatten
        x2 = F.relu(self.fc12(x2))
        x2 = self.fc22(x2)
        
        # concatenate the results of two channels
        x = torch.cat([x1, x2], dim=1) # size n*20
        x = F.relu(x)
        
        x = torch.sigmoid(self.fc3(x)) # size n*2
        
        return x, x1, x2

# MLP + weight sharing + auxiliary loss
class AuxsiameseMLP(nn.Module):
    '''
    basic structure similar to the MLP
    input is splited into two 1*14*14 images for separating training, share the same parameters
    softmax for the auxiliary output layers
    '''
    def __init__(self):
        super(AuxsiameseMLP, self).__init__()
        
        self.fc1 = nn.Linear(196, 160) 
        self.fc2 = nn.Linear(160, 10)
        self.fc3 = nn.Linear(20, 2) 


    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = x1.view(-1,196) # flatten
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        
        # train the second splitted image in second channel
        x2 = x2.view(-1,196) # flatten
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        
        # concatenate the results of two channels
        x = torch.cat([x1, x2], dim=1) # size n*20
        x = F.relu(x)
        
        x = torch.sigmoid(self.fc3(x)) # size n*2
        
        return x, x1, x2









