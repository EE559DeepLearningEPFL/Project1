import torch
# Help functions

# Count the number of parameters
def count_param(model):
    return sum([torch.numel(param) for param in model.parameters()])

# Count number of errors for base models
def compute_nb_errors(model, data_loader):

    nb_data_errors = 0

    for data_input, data_target, data_classes in data_loader:
        output = model(data_input)
        nb_error = torch.sum(torch.argmax(output, dim=1, keepdim=True) != torch.argmax(data_target, dim=1, keepdim=True))
        nb_data_errors += nb_error
        
    return nb_data_errors

# Count number of errors for models with weight sharing
def compute_nb_errors_siamese(model, data_loader):

    nb_data_errors = 0
    for data_input, data_target, data_classes in data_loader:
        data_1, data_2 = data_input.unbind(1)               
        output = model(data_1.unsqueeze(1), data_2.unsqueeze(1))
        nb_error = torch.sum(torch.argmax(output, dim=1, keepdim=True) != torch.argmax(data_target, dim=1, keepdim=True))
        nb_data_errors += nb_error
        
    return nb_data_errors

# Count number of errors for models with Auxiliary loss or both Auxiliary loss and weight sharing
def compute_nb_errors_auxsiamese(model, data_loader):

    nb_data_errors = 0
    for data_input, data_target, data_classes in data_loader:
        data_1, data_2 = data_input.unbind(1)               
        output, aux1, aux2 = model(data_1.unsqueeze(1), data_2.unsqueeze(1))
        nb_error = torch.sum(torch.argmax(output, dim=1, keepdim=True) != torch.argmax(data_target, dim=1, keepdim=True))
        nb_data_errors += nb_error
        
    return nb_data_errors




















