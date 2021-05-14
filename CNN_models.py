# CNN family: CNN, CNN + weight sharing, CNN + auxiliary loss, CNN + weight sharing + auxiliary loss
from torch import nn
from torch.nn import functional as F
import torch

# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)    # size [nb, 32, 12, 12]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # size [nb, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [nb, 32, 6, 6]      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [nb, 64, 2, 2]
        x = x.view(-1, 256) # size [nb, 256]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
# CNN + weight sharing
class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)    # size [nb, 32, 12, 12]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # size [nb, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 2)
        
    def convs(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [nb, 32, 6, 6]      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [nb, 64, 2, 2]
        return x
    
    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256)
        x1 = F.relu((self.fc1(x1)))
        x1 = F.relu(self.fc2(x1))
        
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = F.relu(self.fc2(x2))
        
        x = torch.cat([x1, x2], dim=1)        
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
# CNN + auxiliary loss
class AuxCNN(nn.Module):
    def __init__(self):
        super(AuxCNN, self).__init__()
        self.conv11 = nn.Conv2d(1, 32, kernel_size=3)    # size [nb, 32, 12, 12]
        self.conv21 = nn.Conv2d(32, 64, kernel_size=3)   # size [nb, 64, 4, 4]
        self.fc11 = nn.Linear(256, 200)
        self.fc21 = nn.Linear(200, 10)
        self.conv12 = nn.Conv2d(1, 32, kernel_size=3)    # size [nb, 32, 12, 12]
        self.conv22 = nn.Conv2d(32, 64, kernel_size=3)   # size [nb, 64, 4, 4]
        self.fc12 = nn.Linear(256, 200)
        self.fc22 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 2)
        
    def convs(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [nb, 32, 6, 6]      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [nb, 64, 2, 2]
        return x
    
    def forward(self, x1, x2):
        x1 = F.relu(F.max_pool2d(self.conv11(x1), kernel_size=2)) # size [nb, 32, 6, 6]  
        x1 = F.relu(F.max_pool2d(self.conv21(x1), kernel_size=2)) # size [nb, 64, 2, 2]
        x1 = x1.view(-1, 256)
        x1 = F.relu((self.fc11(x1)))
        x1 = self.fc21(x1)
        aux1 = F.softmax(x1)
        x1 = F.relu(x1)
        
        x2 = F.relu(F.max_pool2d(self.conv12(x2), kernel_size=2)) # size [nb, 32, 6, 6]  
        x2 = F.relu(F.max_pool2d(self.conv22(x2), kernel_size=2)) # size [nb, 64, 2, 2]
        x2 = x2.view(-1, 256)
        x2 = F.relu((self.fc12(x2)))
        x2 = self.fc22(x2)
        aux2 = F.softmax(x2)
        x2 = F.relu(x2)
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.sigmoid(self.fc3(x))
        
        return x, aux1, aux2
    
# CNN + weight sharing + auxiliary loss
class AuxsiameseCNN(nn.Module):
    def __init__(self):
        super(AuxsiameseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)    # size [nb, 32, 12, 12]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # size [nb, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 2)
        
    def convs(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [nb, 32, 6, 6]      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [nb, 64, 2, 2]
        return x
    
    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256)
        x1 = F.relu((self.fc1(x1)))
        x1 = self.fc2(x1)
        aux1 = F.softmax(x1)
        x1 = F.relu(x1)
        
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        aux2 = F.softmax(x2)
        x2 = F.relu(x2)
        
        x = torch.cat([x1, x2], dim=1)       
        x = torch.sigmoid(self.fc3(x))
        
        return x, aux1, aux2






















