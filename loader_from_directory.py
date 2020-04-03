import os
import random
import string
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


torch.cuda.init()
# print(torch.cuda.is_available())
nvcc_args = [
    '-gencode', 'arch=compute_30,code=sm_30',
    '-gencode', 'arch=compute_35,code=sm_35',
    '-gencode', 'arch=compute_37,code=sm_37',
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75'
]

alfabet = list(string.ascii_lowercase)
matrice_encoding = {}

# Cream un dictionar alcatuit din litere si simboluri, pregatit pentru one-hot encoding
for litera in alfabet:
    matrice_encoding[litera] = [0] * 37

# Realizam procesul de one-hot encoding pentru fiecare litera
poz = 0
for litera_curenta in matrice_encoding.keys():
    matrice_encoding[litera_curenta][poz] = 1
    poz += 1

# Simboluri
simboluri = ["space", "number", "period", "comma", "colon", "apostrophe", "hyphen", "semicolon", "question",
           "exclamation", "capitalize"]

# Repetam procedeul de mai sus si pentru simboluri
for simbol in simboluri:
    matrice_encoding[simbol] = [0] * 37

for simbol in simboluri:
    matrice_encoding[simbol][poz] = 1
    poz += 1

data = []

for tgt in os.listdir("dataset"):
    if not tgt == ".DS_Store":
        for folder in os.listdir("dataset/" + tgt + "/Uploaded"):
            if not folder == ".DS_Store":
                for filename in os.listdir("dataset/" + tgt + "/Uploaded/" + folder):
                    if not filename == ".DS_Store":
                        # Obtinem imaginea si denumirea
                        picture = []
                        curr_target = matrice_encoding[tgt]
                        image = Image.open("dataset/" + tgt + "/Uploaded/" + folder + "/" + filename)
                        image = image.convert('RGB')
                        image = np.array(image)
                        # Redimensionam imaginea la 28x28x3
                        image = cv2.resize(image, (28, 28))
                        # Normalizare 0-1
                        image = image.astype(np.float32) / 255.0
                        image = torch.from_numpy(image).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                        picture.append(image)
                        # Convertim datele intr-un long tensor
                        curr_target = torch.Tensor([curr_target])
                        picture.append(curr_target)
                        # Concatenam datele
                        data.append(picture)

# Cream un dictionar cu toate caracterele
caractere = alfabet + simboluri

index2char = {}
poz = 0
for caracter in caractere:
    index2char[poz] = caracter
    poz += 1


# find the number of each character in a dataset
def num_chars(dataset, index2char):
    chars = {}
    for _, label in dataset:
        char = index2char[int(torch.argmax(label))]
        # update
        if char in chars:
            chars[char] += 1
        # initialize
        else:
            chars[char] = 1
    return chars


# Create dataloader objects

# Amestecam aleator datele
random.shuffle(data)

# Cream loturi pentru invatare, testare si validare
batch_size_train = 30
batch_size_test = 30
batch_size_validation = 30

# 1600 pentru invatare
train_dataset = data[:22000]
# 212 pentru testare
test_dataset = data[22000:24400]
# 212 pentru validare
validation_dataset = data[24400:]

# create the dataloader objects
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size_validation,
                                                shuffle=True)

# to check if a dataset is missing a char
test_chars = num_chars(test_dataset, index2char)

num = 0
for caracter in caractere:
    if caracter in test_chars:
        num += 1
    else:
        break
print(num)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            # 3x28x28
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # batch normalization
            # nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True), 
            # 16x28x28
            nn.MaxPool2d(kernel_size=2),
            # 16x14x14
            nn.LeakyReLU()
        )
        # 16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # batch normalization
            # nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), 
            # 32x14x14
            nn.MaxPool2d(kernel_size=2),
            # 32x7x7
            nn.LeakyReLU()
        )
        # linearly 
        self.block3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),
            # batch normalization
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 37)
        )
        # 1x37

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        # flatten the dataset
        out = out.view(-1, 32 * 7 * 7)
        out = self.block3(out)
        return out


# convolutional neural network model
cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
model = CNN()
model.to(device)

# print summary of the neural network model to check if everything is fine. 
print(model)
print("# parameter: ", sum([param.nelement() for param in model.parameters()]))

# setting the learning rate
learning_rate = 1e-4

# Using a variable to store the cross entropy method
criterion = nn.CrossEntropyLoss().to(device)

# Using a variable to store the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# list of all train_losses 
train_losses = []

# list of all validation losses 
validation_losses = []

# for loop that iterates over all the epochs
num_epochs = 200
for epoch in range(num_epochs):

    # variables to store/keep track of the loss and number of iterations
    train_loss = 0
    num_iter_train = 0

    # train the model
    model.train()

    # Iterate over train_loader
    for i, (images, labels) in enumerate(train_loader):
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Zero the gradient buffer
        # resets the gradient after each epoch so that the gradients don't add up
        optimizer.zero_grad()

        # Forward, get output
        outputs = model(images)

        # convert the labels from one hot encoding vectors into integer values
        cuda_ = "cuda:0"
        device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1).to(device)

        # calculate training loss
        loss = criterion(outputs, y_true).to(device)

        # Backward (computes all the gradients)
        loss.backward()

        # Optimize
        # loops through all parameters and updates weights by using the gradients 
        # takes steps backwards to optimize (to reach the minimum weight)
        optimizer.step()
        # update the training loss and number of iterations
        train_loss += loss.data
        num_iter_train += 1

    print('Epoch: {}'.format(epoch + 1))
    print('Training Loss: {:.4f}'.format(train_loss / num_iter_train))
    # append training loss over all the epochs
    train_losses.append(train_loss / num_iter_train)

    # evaluate the model
    model.eval()

    # variables to store/keep track of the loss and number of iterations
    validation_loss = 0
    num_iter_validation = 0

    # Iterate over validation_loader
    for i, (images, labels) in enumerate(validation_loader):
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Forward, get output
        outputs = model(images)

        # convert the labels from one hot encoding vectors to integer values
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1).to(device)

        # calculate the validation loss
        loss = criterion(outputs, y_true).to(device)

        # update the training loss and number of iterations
        validation_loss += loss.data
        num_iter_validation += 1

    print('Validation Loss: {:.4f}'.format(validation_loss / num_iter_validation))
    # append all validation_losses over all the epochs
    validation_losses.append(validation_loss / num_iter_validation)

    num_iter_test = 0
    correct = 0

    # Iterate over test_loader
    for images, labels in test_loader:
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Forward
        outputs = model(images)

        # convert the labels from one hot encoding vectors into integer values 
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1).to(device)

        # find the index of the prediction
        y_pred = torch.argmax(outputs, 1).type('torch.FloatTensor').to(device)

        # convert to FloatTensor
        y_true = y_true.type('torch.FloatTensor').to(device)

        # find the mean difference of the comparisons
        correct += torch.sum(torch.eq(y_true, y_pred).type('torch.FloatTensor')).to(device)
    x=(correct / len(test_dataset) * 100)
    # print('Asta e x-ul meu %.4f'%x)
    nume=str('{:.4f}'.format(x))+"model_1_final__.pth"
    if(x>99.89) :
        torch.save(model.state_dict(),nume)
        print('Saved!')
    print('Accuracy on the test set: {:.4f}%'.format(correct / len(test_dataset) * 100))
    print()


# learning curve function
def plot_learning_curve(train_losses, validation_losses):
    # plot the training and validation losses
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.plot(train_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.legend(loc=1)
    plt.show()


# plot the learning curve
plt.title("Learning Curve (Loss vs Number of Epochs)")
plot_learning_curve(train_losses, validation_losses)
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]


# torch.save(model.state_dict(), "model_1_final_1.pth")
