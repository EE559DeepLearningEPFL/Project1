# Main result
import torch
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue
import time

from MLP_models import *
from CNN_models import *
from Resnet_models import *
from helper import *
from data_loader import *
from train import *

# Print out the parameters of each model
model_1 = MLP()
model_2 = SiameseMLP()
model_3 = AuxMLP()
model_4 = AuxsiameseMLP()
model_5 = CNN()
model_6 = SiameseCNN()
model_7 = AuxCNN()
model_8 = AuxsiameseCNN()
model_9 = ResNet()
model_10 = SiameseResNet()
model_11 = AuxResNet()
model_12 = AuxsiameseResNet()

models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12]
for i in range(len(models)):
    print('The number of parameters in model_%d is %d' %
                  (i+1, count_param(models[i])))
    
    
    
# Final result

# Models with optimal parameters
# [model name, if weight sharing, if auxiliary loss, learning rate, batch size, auxiliary coefficient: alpha]
Train_final_model = [[MLP, False, False, 5e-4, 8, 0],
              [SiameseMLP, True, False, 5e-3, 8, 0],
              [AuxMLP, False, True, 5e-3, 16, 0.9],
              [AuxsiameseMLP, True, True, 5e-3, 8, 0.7],
              [CNN, False, False, 5e-4, 16, 0],
              [SiameseCNN, True, False, 5e-3, 32, 0],
              [AuxCNN, False, True, 5e-4, 8, 1.0],
              [AuxsiameseCNN, True, True, 5e-3, 32, 0.6],
              [ResNet, False, False, 1e-3, 16, 0],
              [SiameseResNet, True, False, 1e-3, 16, 0],
              [AuxResNet, False, True, 1e-3, 64, 0.7],
              [AuxsiameseResNet, True, True, 1e-3, 16, 0.8]]

# Print out the final results
def results(Train_final_model): 
    '''
    every model is trained 10 times to get average accuracy and std
    '''
    loss_total = [] # list to store the training loss changes in each model
    index = 0 # label the number of model
    for models in Train_final_model:
        times = [] # list to store training time
        accuracies = [] # list to store accuracy
        losses = torch.empty((10,25)) # list to store loss

        for i in range(10,20):
            train_loader, test_loader = load_data(N=1000, batch_size=models[4], seed=i) # load data
            time1 = time.perf_counter()

            model = models[0]() # assign model
            model.to(device) # move model to device
            losses[i-10,:] = torch.tensor(train(model, train_loader, models[3], 0, 25, verbose=False, siamese=models[1], aux=models[2], alpha=models[5]))
            time2 = time.perf_counter()
            times.append(time2 - time1)

            tr_accuracy, te_accuracy = accu(model, train_loader, test_loader, siamese=models[1], aux=models[2]) # calculate accuracy

            accuracies.append(te_accuracy)
        loss_total.append(losses)
        index += 1
    # Print out results
        print('For optimal model_%d, Mean accuracy: %.3f, Std: %.3f, Mean time: %.3f, Std: %.3f' %(index, torch.tensor(accuracies).mean(), torch.tensor(accuracies).std(), torch.tensor(times).mean(), torch.tensor(times).std()))    

        
        
# Final results        
results(Train_final_model)    





