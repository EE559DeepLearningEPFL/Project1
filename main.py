# Main result

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
Train_final_model = [[MLP, False, False, 5e-4, 8, 0],
              [SiameseMLP, True, False, 5e-4, 8, 0],
              [AuxMLP, False, True, 5e-3, 16, 0.9],
              [AuxsiameseMLP, True, True, 5e-3, 8, 0.7],
              [CNN, False, False, 5e-4, 16, 0],
              [SiameseCNN, True, False, 5e-4, 16, 0],
              [AuxCNN, False, True, 5e-4, 8, 1.0],
              [AuxsiameseCNN, True, True, 5e-3, 32, 0.6],
              [ResNet, False, False, 1e-3, 32, 0],
              [SiameseResNet, True, False, 5e-3, 32, 0],
              [AuxResNet, False, True, 5e-3, 32, 0.6],
              [AuxsiameseResNet, True, True, 5e-3, 32, 0.6]]
    
def results(Train_final_model):    
    loss_total = []
    index = 0
    for models in Train_final_model:
        times = []
        accuracies = []
        losses = torch.empty((10,25))

        for i in range(10):
            train_loader, test_loader = load_data(N=1000, batch_size=models[4], seed=i)
            time1 = time.perf_counter()

            model = models[0]()
            model.to(device)
            losses[i,:] = torch.tensor(train(model, train_loader, models[3], 0, 25, verbose=False, siamese=models[1], aux=models[2], alpha=models[5]))
            time2 = time.perf_counter()
            times.append(time2 - time1)

            tr_accuracy, te_accuracy = accu(model, train_loader, test_loader, siamese=models[1], aux=models[2])

            accuracies.append(te_accuracy)
        loss_total.append(losses)
        index += 1
    
        print('For optimal model_%d, Mean accuracy: %.3f, Std: %.3f, Mean time: %.3f, Std: %.3f' %(index, torch.tensor(accuracies).mean(), torch.tensor(accuracies).std(), torch.tensor(times).mean(), torch.tensor(times).std()))    

        
        
        
results(Train_final_model)    





