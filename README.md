# Project1
Members: Menghe Jin, Yueqing Shen, Jiaan Zhu

Models are in the CNN_models.py, MLP_models.py, Resnet_models.py
In the submitted version, we use softmax before calculating the crossentropy loss. However, in pytorch, softmax is already integrated in nn.CrossEntropyLoss, we remove softmax in the current repo. No big differences, slight accuracy improvement(around 0.01) in the new version of code.

Run test.py to get the final results
