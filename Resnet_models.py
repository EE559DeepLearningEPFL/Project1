# ResNet family: , ResNet + weight sharing, ResNet + auxiliary loss, ResNet + weight sharing + auxiliary loss
from torch import nn
from torch.nn import functional as F
import torch

# ResNetBlock with skip-connection and batch normalization
class ResNetBlock(nn.Module):
    '''
    code reused from practical section 6
    '''
    def __init__(self, nb_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = y + x
        y = F.relu(y)

        return y
    
# ResNet 
class ResNet(nn.Module):
    '''
    nb_residual_blocks = 4,
    input_channels = 2,
    nb_channels = 32,
    kernel_size = 3,
    nb_classes = 2
    '''
    def __init__(self, nb_residual_blocks = 4, input_channels = 2, nb_channels = 32, kernel_size = 3, nb_classes = 2):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size)
              for _ in range(nb_residual_blocks))
        )

        self.fc = nn.Linear(nb_channels*9, nb_classes) # input size is nb_channels*(14//4)^2 = nb_channels*9
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.resnet_blocks(x)
        x = F.avg_pool2d(x, 4).view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

    
# ResNet + weight sharing
class SiameseResNet(nn.Module):
    '''
    nb_residual_blocks = 4,
    input_channels = 1,
    nb_channels = 32,
    kernel_size = 3,
    nb_classes = 2
    input is splited into two 1*14*14 images for separating training, share the same parameters
    '''
    def __init__(self, nb_residual_blocks = 4, input_channels = 1, nb_channels = 32, kernel_size = 3, nb_classes = 2):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size)
              for _ in range(nb_residual_blocks))
        )

        self.fc = nn.Linear(20, nb_classes)
        self.fc1 = nn.Linear(nb_channels*9, 10) # input size is nb_channels*(14//4)^2 = nb_channels*9
        
        
    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = F.relu(self.bn(self.conv(x1)))
        x1 = self.resnet_blocks(x1)
        x1 = F.avg_pool2d(x1, 4).view(x1.size(0), -1) # flatten
        x1 = F.relu(self.fc1(x1))
        
        # train the second splitted image in second channel
        x2 = F.relu(self.bn(self.conv(x2)))
        x2 = self.resnet_blocks(x2)
        x2 = F.avg_pool2d(x2, 4).view(x2.size(0), -1) # flatten
        x2 = F.relu(self.fc1(x2))
        
        x = torch.cat([x1, x2], dim=1)   
        x = torch.sigmoid(self.fc(x))
        
        return x
    

# ResNet + auxiliary loss
class AuxResNet(nn.Module):
    '''
    nb_residual_blocks = 4,
    input_channels = 1,
    nb_channels = 32,
    kernel_size = 3,
    nb_classes = 2
    input is splited into two 1*14*14 images for separating training, use different parameters
    
    '''
    def __init__(self, nb_residual_blocks = 4, input_channels = 1, nb_channels = 32, kernel_size = 3, nb_classes = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(input_channels, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.bn2 = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks1 = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size)
              for _ in range(nb_residual_blocks))
        )
        self.resnet_blocks2 = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size)
              for _ in range(nb_residual_blocks))
        )
        
        self.fc1 = nn.Linear(nb_channels*9, 10) # input size is nb_channels*(14//4)^2 = nb_channels*9
        self.fc = nn.Linear(20, nb_classes)
        
    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = self.resnet_blocks1(x1)
        x1 = F.avg_pool2d(x1, 4).view(x1.size(0), -1) # flatten
        x1 = self.fc1(x1)
        # softmax for the auxiliary output layer of first channel
        aux1 = F.softmax(x1, dim=1)
        x1 = F.relu(x1)
        
        # train the second splitted image in second channel
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2 = self.resnet_blocks2(x2)
        x2 = F.avg_pool2d(x2, 4).view(x2.size(0), -1) # flatten
        x2 = self.fc1(x2)
        # softmax for the auxiliary output layer of second channel
        aux2 = F.softmax(x2, dim=1)
        x2 = F.relu(x2)
        
        # concatenate the results of two channels
        x = torch.cat([x1, x2], dim=1)
        
        x = torch.sigmoid(self.fc(x))    
        
        return x, aux1, aux2
    

# ResNet + weight sharing + auxiliary loss
class AuxsiameseResNet(nn.Module):
    '''
    nb_residual_blocks = 4,
    input_channels = 1,
    nb_channels = 32,
    kernel_size = 3,
    nb_classes = 2
    input is splited into two 1*14*14 images for separating training, share the same parameters
    softmax for the auxiliary output layers
    '''
    def __init__(self, nb_residual_blocks = 4, input_channels = 1, nb_channels = 32, kernel_size = 3, nb_classes = 2):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size)
              for _ in range(nb_residual_blocks))
        )

        self.fc1 = nn.Linear(nb_channels*9, 10) # input size is nb_channels*(14//4)^2 = nb_channels*9
        self.fc = nn.Linear(20, nb_classes)
        
    def forward(self, x1, x2):
        # train the first splitted image in first channel
        x1 = F.relu(self.bn(self.conv(x1)))
        x1 = self.resnet_blocks(x1)
        x1 = F.avg_pool2d(x1, 4).view(x1.size(0), -1) # flatten
        x1 = self.fc1(x1)
        # softmax for the auxiliary output layer of first channel
        aux1 = F.softmax(x1, dim=1)
        x1 = F.relu(x1)
        
        # train the second splitted image in second channel
        x2 = F.relu(self.bn(self.conv(x2)))
        x2 = self.resnet_blocks(x2)
        x2 = F.avg_pool2d(x2, 4).view(x2.size(0), -1) # flatten
        x2 = self.fc1(x2)
        # softmax for the auxiliary output layer of second channel
        aux2 = F.softmax(x2, dim=1)
        x2 = F.relu(x2)
        
        # concatenate the results of two channels
        x = torch.cat([x1, x2], dim=1)
        
        x = torch.sigmoid(self.fc(x)) 
        
        return x, aux1, aux2
    
    
    
    
    
    
    
    
