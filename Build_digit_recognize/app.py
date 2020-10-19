
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import torch
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from time import time
from time import sleep
from PIL import ImageTk, Image, ImageDraw, ImageOps, ImageFilter
from torchvision import datasets, transforms
from torch import nn, optim
from pathlib import Path
import os
# from google.colab import drive
import warnings


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
width = 400
height = 450
center = height//2
white = (255, 255, 255)
green = (0,128,0)
black = (0,0,0)

dir1 = Path(__file__).parent.absolute()

def open_iamge_path():
    path = filedialog.askopenfilename(
        parent=root, initialdir='C:/Tutorial',
        title='Choose file',
        filetypes=[('png images', '.png'),
                   ('jpg images', '.jpg'),]
        )

    """ Show image by click Select button"""
    print(path)
    try:
        load = Image.open(path)
    except IOError:
        print("Unable to load image")
        sys.exit(1)
    
    predict(path)
    sys.exit()

def training():
    warnings.filterwarnings("ignore")

    model = ( str(dir1) + '/my_mnist_model.pt')
    if os.path.exists(model):
        result = messagebox.askquestion("Delete","Do you want to Delete")
        if(result == 'yes'):
            label['text'] = "Wait trainning done"
            sleep(5)
            print("Delete file")
            os.remove(model)
        else:
            sys.exit()

    #Wait 5s after that start trainning
    sleep(5)
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    # Download and load the training data
    trainset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=True, transform=transform)
    valset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Layer details for the neural network
    input_size = 784
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
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images)
    loss = criterion(logps, labels)
    loss.backward()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    images, labels = next(iter(trainloader))
    images.resize_(64, 784)
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
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

    label['text'] = "Predicted Digit ="
    # Save model
    print("Save Model: ./my_mnist_model.pt")
    torch.save(model, './my_mnist_model.pt') 
    #messagebox.showinfo( "GUI information", "Save Model: ./my_mnist_model.pt")


def predict(path_image):
    model = torch.load( str(dir1) + '/my_mnist_model.pt')
    # if(path_image is None):
    #     img = Image.open(str(dir1) + '/image.png')
    # else:
    img = Image.open(str(dir1) + '/image.png')
    img = img.resize((28,28),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    img = ImageOps.grayscale(img)
    img = transform(img)
    im2arr = np.array(img)
    # im2arr = np.array([(255 - x)/255. for x in im2arr])
    tensor1 = torch.from_numpy(im2arr)
    with torch.no_grad():
        logps = model(tensor1.view(1,784).float())

    # Output of the network are log-proba  bilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    # print("Predicted Digit =", probab.index(max(probab)))
    a = "Predicted Digit =" + str(probab.index(max(probab)))
    # label = Label(root, text = a)
    label['text'] = a
    view_classify(img.view(1, 28, 28), ps)
    

def save():
    filename = str(dir1) + '/image.png'
    image1.save(filename, as_gray = True)
    predict(filename)

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    # cv.create_oval(x1, y1, x2, y2, fill="white",width=20)
    cv.create_rectangle(x1, y1, x2, y2, fill="white",width=20)
    draw.rectangle([x1, y1, x2, y2],fill="white",width=10)
    # draw.line([x1, y1, x2, y2],fill="white",width=10)

def erase():
    cv.delete("all")
    plt.close()
    draw.rectangle((0, 0, 500, 500), fill=(0, 0, 0, 0))

def help_button():
    help_message = """
        Verion 0.0.1 - Design by bpham
        How to use this GUI:
        1. Draw a random Digit [0 -> 9]
        2. Press Predict and wait until result ready to show.
        3. Result will show on old windows, and new windowns for debug.
    """
    messagebox.showinfo( "GUI information", help_message)



root = Tk()
root.title('Digit Recogniser')

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg = 'white')
cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

image1 = PIL.Image.new("RGB", (width, height), black)
draw = ImageDraw.Draw(image1)

label = Label(root)
label = Label(root, text = "Predicted Digit = ")
label.pack()

button_erase=Button(root, text="Erase",width = 8, command=erase, bg = '#6f8396')
button_erase.pack(side= LEFT)

button_save=Button(root, text="Predict",width = 8, command=save, bg = '#6f8396')
button_save.pack(side = LEFT)

button_select=Button(root, text="Select",width = 8, command=open_iamge_path, bg = '#6f8396')
button_select.pack(side = 'right')

button_help=Button(root, text="Help",width = 8, command=help_button, bg = '#6f8396')
button_help.pack(side = 'right')

button_train=Button(root, text="Trainning",width = 8, command=training, bg = '#6f8396')
button_train.pack(side = 'right')



root.mainloop()
