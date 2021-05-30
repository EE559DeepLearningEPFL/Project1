# CNN family: CNN, CNN + weight sharing, CNN + auxiliary loss, CNN + weight sharing + auxiliary loss
from torch import nn
from torch.nn import functional as F
import torch

# CNN
class CNN(nn.Module):
    '''
    conv1, conv2
        two convolution layers.
    fc1, fc2
        two fully connected layers.
    fc3
        output layer
    relu
        activation function for hidden layers
    sigmoid
        activation function for output layer
    '''
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)    # size [n, 32, 12, 12], n is the number of samples
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # size [n, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [n, 32, 6, 6]      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [n, 64, 2, 2]
        x = x.view(-1, 256) # size [nb, 256]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
# CNN + weight sharing
class SiameseCNN(nn.Module):
    '''
    basic structure similar to the CNN
    input is splited into two 1*14*14 images for separating training, share the same parameters
    '''
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)    # size [n, 32, 12, 12]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # size [n, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 2)
        
    def convs(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [n, 32, 6, 6]      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [n, 64, 2, 2]
        return x
    
    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256) # flatten
        x1 = F.relu((self.fc1(x1)))
        x1 = F.relu(self.fc2(x1))
        
        # train the second splitted image in second channel
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256) # flatten
        x2 = F.relu(self.fc1(x2))
        x2 = F.relu(self.fc2(x2))
        
        x = torch.cat([x1, x2], dim=1)        
        x = torch.sigmoid(self.fc3(x)) 
        
        return x
    
# CNN + auxiliary loss
class AuxCNN(nn.Module):
    '''
    basic structure similar to the CNN
    input is splited into two 1*14*14 images for separating training, use different parameters
    softmax for the auxiliary output layers
    '''
    def __init__(self):
        super(AuxCNN, self).__init__()
        self.conv11 = nn.Conv2d(1, 32, kernel_size=3)    # conv1 for first channel, size [n, 32, 12, 12]
        self.conv21 = nn.Conv2d(32, 64, kernel_size=3)   # conv2 for first channel, size [n, 64, 4, 4]
        self.fc11 = nn.Linear(256, 200) # fc1 for first channel
        self.fc21 = nn.Linear(200, 10) # fc2 for first channel
        self.conv12 = nn.Conv2d(1, 32, kernel_size=3)    # conv1 for second channel, size [n, 32, 12, 12]
        self.conv22 = nn.Conv2d(32, 64, kernel_size=3)   # conv2 for second channel, size [n, 64, 4, 4]
        self.fc12 = nn.Linear(256, 200) # fc1 for second channel
        self.fc22 = nn.Linear(200, 10) # fc2 for second channel
        self.fc3 = nn.Linear(20, 2) 
        
    def convs(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [n, 32, 6, 6]      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [n, 64, 2, 2]
        return x
    
    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = F.relu(F.max_pool2d(self.conv11(x1), kernel_size=2)) # size [n, 32, 6, 6]  
        x1 = F.relu(F.max_pool2d(self.conv21(x1), kernel_size=2)) # size [n, 64, 2, 2]
        x1 = x1.view(-1, 256) # flatten
        x1 = F.relu((self.fc11(x1)))
        x1 = self.fc21(x1)
        
        # train the second splitted image in second channel
        x2 = F.relu(F.max_pool2d(self.conv12(x2), kernel_size=2)) # size [n, 32, 6, 6]  
        x2 = F.relu(F.max_pool2d(self.conv22(x2), kernel_size=2)) # size [n, 64, 2, 2]
        x2 = x2.view(-1, 256) # flatten
        x2 = F.relu((self.fc12(x2)))
        x2 = self.fc22(x2)
        
        # concatenate the results of two channels
        x = torch.cat([x1, x2], dim=1) # size n*20
        x = F.relu(x)
                
        x = torch.sigmoid(self.fc3(x)) # size n*2
        
        return x, x1, x2
    
# CNN + weight sharing + auxiliary loss
class AuxsiameseCNN(nn.Module):
    '''
    basic structure similar to the CNN
    input is splited into two 1*14*14 images for separating training, share the same parameters
    softmax for the auxiliary output layers
    '''
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
        # train the first splitted image in first channel
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256) # flatten
        x1 = F.relu((self.fc1(x1)))
        x1 = self.fc2(x1)
        
        # train the second splitted image in second channel
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256) # flatten
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        
        # concatenate the results of two channels
        x = torch.cat([x1, x2], dim=1) # size n*20   
        x = F.relu(x)
        
        x = torch.sigmoid(self.fc3(x)) # size n*2
        
        return x, x1, x2






















