# train.py
# Train function and accuracy function
from helper import *
import torch
from torch import nn
from torch.nn import functional as F
from data_loader import *
import dlc_practical_prologue as prologue

# Train function
def train(model, train_loader, eta, decay, n_epochs=25, verbose=False, siamese=False, aux=False, alpha = 0):
    '''
    model: learning model
    train_loader: train loader
    eta: parameter of learning rate
    decay: parameter of weight delay
    n_epoch: parameter pf number of epochs
    verbose: if print total loss of each epoch
    siamese: if use weight sharing
    aux: if use auxiliary loss
    alpha: parameter of coefficient of auxiliary loss
    '''
    
    binary_crit = torch.nn.BCELoss() # BCEloss for total loss
    aux_crit = torch.nn.CrossEntropyLoss() # Cross entropy loss for auxiliary loss
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=decay) # Adam optimizer
    
    # List to store the train accuracy of each epoch
    tr_losses = []


    for e in range(n_epochs):
        # Reset training loss
        tr_loss = 0

        # Training model
        model.train()

        for train_input, train_target, train_classes in iter(train_loader):
            
            # Forward pass
            if aux == True: # For all models with auxiliary loss
                train_1, train_2 = train_input.unbind(1)
                output, aux1, aux2 = model(train_1.unsqueeze(1), train_2.unsqueeze(1))
                
            elif siamese == True: # For models with weight sharing only
                train_1, train_2 = train_input.unbind(1)
                output = model(train_1.unsqueeze(1), train_2.unsqueeze(1))
                
            else: # For models without auxiliary loss and weight sharing
                output = model(train_input)
                
            # Binary classification loss
            binary_loss = binary_crit(output, train_target.float())
            total_loss = binary_loss
            
            # Auxiliary loss
            if aux == True:

                aux_loss1 = aux_crit(aux1, train_classes[:,0])
                aux_loss2 = aux_crit(aux2, train_classes[:,1])
                aux_loss = aux_loss1 + aux_loss2
                
                # Total loss = Binary loss + aux loss * alpha
                total_loss = binary_loss + aux_loss * alpha
            
            # training loss
            tr_loss += total_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Collect training loss of each epoch
        tr_losses.append(tr_loss)

        if verbose:
            print('Epoch %d/%d, total loss: %.3f' %
                  (e+1, n_epochs, tr_loss))
    return tr_losses


# accuracy function
def accu(model, train_loader, test_loader, siamese = False, aux = False):
    '''
    function for calculating the train accuracy and test accuracy
    '''
    if aux == True: # For all models with auxiliary loss
        tr_accuracy = 1 - compute_nb_errors_auxsiamese(model, train_loader)/1000
        te_accuracy = 1 - compute_nb_errors_auxsiamese(model, test_loader)/1000
        
    elif siamese == True: # For models with weight sharing only
        tr_accuracy = 1 - compute_nb_errors_siamese(model, train_loader)/1000
        te_accuracy = 1 - compute_nb_errors_siamese(model, test_loader)/1000
        
    else: # For models without auxiliary loss and weight sharing
        tr_accuracy = 1 - compute_nb_errors(model, train_loader)/1000
        te_accuracy = 1 - compute_nb_errors(model, test_loader)/1000
            
    return tr_accuracy, te_accuracy

















