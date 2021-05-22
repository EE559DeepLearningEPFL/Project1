# optimization.py
import torch
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue

from MLP_models import *
from CNN_models import *
from Resnet_models import *
from helper import *
from data_loader import *
from train import *

# initial models for tuning learning rate and batch size
# [model name, if weight sharing, if auxiliary loss]
Train_model = [[MLP, False, False],
              [SiameseMLP, True, False],
              [AuxMLP, False, True],
              [AuxsiameseMLP, True, True],
              [CNN, False, False],
              [SiameseCNN, True, False],
              [AuxCNN, False, True],
              [AuxsiameseCNN, True, True],
              [ResNet, False, False],
              [SiameseResNet, True, False],
              [AuxResNet, False, True],
              [AuxsiameseResNet, True, True]]


def lr_bs_tuning(Train_model):
    '''
    tuning the learning rate and batch size
    '''
    accuracy = [] # list for accuracy
    std = [] # list for std
    model_number = 0 # index of model
    
    for models in Train_model:
        gammas = [5e-3, 1e-3, 5e-4, 1e-4] # learning rates
        batch_sizes = [8, 16, 32, 64, 128] # batch sizes
        
        test_accuracies = torch.empty((len(gammas), len(batch_sizes))) # store accuracy
        test_stds = torch.empty((len(gammas), len(batch_sizes))) # store std

        for j in range(len(gammas)):
            for k in range(len(batch_sizes)):
                accurate = []
                
                # train 10 times for each condition
                for i in range(10):
                    model = models[0]()
                    model.to(device)
                    train_loader, test_loader = load_data(N=1000, batch_size=batch_sizes[k], seed=i)
                    loss = train(model, train_loader, gammas[j], 0, 25, verbose=False, siamese=models[1], aux=models[2])
                    tr_accuracy, te_accuracy = accu(model, train_loader, test_loader, siamese=models[1], aux=models[2])
                    accurate.append(te_accuracy)
                    
                test_accuracies[j,k] =  torch.FloatTensor(accurate).mean()
                test_stds[j,k] =  torch.FloatTensor(accurate).std()
                
        accuracy.append(test_accuracies)
        std.append(test_stds)
        
        max_index = test_accuracies.argmax() # looking for the maximum accuracy
        
        model_number += 1
        # Print out results
        print('The optimal parameters for model_%d: learning rate %.5f, batch size: %d' %(model_number, gammas[(max_index)//5], batch_sizes[(max_index+1)%5-1]))
        
# models for tuning auxiliary loss, learning rate and batch size based on the previous tuning step
# [model name, if weight sharing, if auxiliary loss, learning rate, batch size]
Train_auxmodel = [[AuxMLP, False, True, 5e-3, 16],
              [AuxsiameseMLP, True, True, 5e-3, 8],
              [AuxCNN, False, True, 5e-4, 8],
              [AuxsiameseCNN, True, True, 5e-3, 32],
              [AuxResNet, False, True, 1e-3, 8],
              [AuxsiameseResNet, True, True, 5e-3, 8]]



def alpha_tuning(Train_auxmodel):
    '''
    tuning the auxiliary loss coefficience alpha
    '''
    accuracy_aux = [] # list for accuracy
    std_aux = [] #list for std
    model_number_aux = 0 # index of model
    
    for models in Train_auxmodel:
        test_accuracies_aux = torch.empty((1, 11)) # store accuracy
        test_stds_aux = torch.empty((1,11)) # store std
        
        for j in range(11):
            accurate_aux = []
            
            for i in range(10):
                model = models[0]()
                model.to(device)
            
                train_loader, test_loader = load_data(N=1000, batch_size = models[4], seed=i)
                # alpha = j/10
                loss = train(model, train_loader, models[3], 0, 25, verbose=False, siamese=models[1], aux=models[2], alpha = j/10)
                tr_accuracy, te_accuracy = accu(model, train_loader, test_loader, siamese=models[1], aux=models[2])
                accurate_aux.append(te_accuracy)
            
            test_accuracies_aux[0,j] =  torch.FloatTensor(accurate_aux).mean()
            test_stds_aux[0,j] =  torch.FloatTensor(accurate_aux).std()
            
        accuracy_aux.append(test_accuracies_aux)
        std_aux.append(test_stds_aux)
        
        max_index = test_accuracies_aux.argmax() # looking for maximum accuracy
        model_number_aux += 1
        # Print out results
        print('The optimal alpha for aux_model_%d is %.2f' %(model_number_aux, max_index/10))
        

        
lr_bs_tuning(Train_model)

alpha_tuning(Train_model)
        
        
        
        
        
        
        
        
        