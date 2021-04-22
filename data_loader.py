# data_loader.py
from torch.utils.data import TensorDataset, DataLoader

def load_data(N=1000, batch_size=50, seed=42):
    # Load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    
    # Normalize data
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)
    
    # Generate dataset
    train_data = TensorDataset(train_input, train_target, train_classes)
    test_data = TensorDataset(test_input, test_target, test_classes)
    
    # For reproducibility
    torch.manual_seed(seed)
    
    # Generate data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, test_loader