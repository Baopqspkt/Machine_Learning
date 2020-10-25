# https://github.com/amitrajitbose/handwritten-digit-recognition/blob/master/handwritten_digit_recognition_CPU.ipynb?fbclid=IwAR2mXFvJQxkyAbPTe1grJ5fThWbjFsLfb0lS5QJlEHaW9hqjzZOQbtuB4z4

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
import os
# from google.colab import drive
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from PIL import Image
import warnings

batch_size_test = 64 #https://viblo.asia/p/lam-quen-voi-pytorch-phan-2-bai-toan-phan-loai-va-deeplearning-924lJDyXKPM
learning_rate = 0.001

def training():
    print("TRAINNING MODEL ")
    warnings.filterwarnings("ignore")
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    # Download and load the training data
    trainset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_test, shuffle=True)  

    valset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=False, transform=transform)
    validadtion_iamge = torch.utils.data.DataLoader(valset, batch_size=batch_size_test, shuffle=True)# Validation data

    # Layer details for the neural network
    input_size = 784 # 28 * 28 = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    time0 = time()
    epochs = 50
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Training pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
        
            #This is where the model learns by backpropagating
            loss.backward()
        
            #And optimizes its weights here
            optimizer.step()
        
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    # Save model
    print("Save Model: ./my_mnist_model.pt")
    torch.save(model, './my_mnist_model.pt') 

    ## Validation 
    correct_count, all_count = 0, 0
    for images,labels in validadtion_iamge:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))
training()
