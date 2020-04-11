import os
import random
import string
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from final_model.resources import plot_cmatrix

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = []

for tgt in os.listdir("dataset"):
    for folder in os.listdir("dataset/" + tgt + "/Uploaded"):
        for filename in os.listdir("dataset/" + tgt + "/Uploaded/" + folder):
            # Obtinem imaginea si o convertim in binar
            picture = []
            curr_target = matrice_encoding[tgt]
            image = Image.open("dataset/" + tgt + "/Uploaded/" + folder + "/" + filename)
            image = image.convert('RGB')
            image = np.array(image)
            # Redimensionam imaginea la 28x28x3
            image = cv2.resize(image, (28, 28))
            # Normalizare 0-1
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).cuda()
            picture.append(image)
            # Convertim echivalentul cuvantului in binar ca tensor si apoi il appenduim la picture
            curr_target = torch.Tensor([curr_target]).cuda()
            picture.append(curr_target)
            # Astfel fiecare pozitie din vectorul data contine 2 atribute: imaginea ca tensor si numele acesteia tensor
            data.append(picture)

# Cream un dictionar cu toate caracterele
caractere = alfabet + simboluri

index2char = {}
poz = 0
for caracter in caractere:
    index2char[poz] = caracter
    poz += 1


# Frecventa caracterelor in dataset
def num_chars(dataset, index2char):
    chars = {}
    for _, label in dataset:
        char = index2char[int(torch.argmax(label).to(device))]
        # update
        if char in chars:
            chars[char] += 1
        # initialize
        else:
            chars[char] = 1
    return chars


random.shuffle(data)

# Cream loturi pentru invatare, testare si validare
batch_size_train = 30
batch_size_test = 30
batch_size_validation = 30

train_dataset = data[:22000]
test_dataset = data[22000:24400]
validation_dataset = data[24400:]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
# train_loader1 are parametrul Shuffle setat pe False pentru a ajuta la alcatuirea unei matrici de confuzie
train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size_validation,
                                                shuffle=True)


# Verificam integritatea datasetului dpdv al nr.de caractere
# test_chars = num_chars(test_dataset, index2char)
#
# num = 0
# summ = 0
# for caracter in caractere:
#     if caracter in test_chars:
#         summ += test_chars[caracter]
#         num += 1
#     else:
#         break
# print(num)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            # 3x28x28
            nn.Conv2d(in_channels=3,
                      out_channels=20,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 16x28x28
            nn.MaxPool2d(kernel_size=2),
            # 16x14x14
            nn.LeakyReLU()
        )
        # 16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=20,
                      out_channels=40,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 32x14x14
            nn.MaxPool2d(kernel_size=2),
            # 32x7x7
            nn.LeakyReLU()
        )
        # linearly 
        self.block3 = nn.Sequential(
            nn.Linear(40 * 7 * 7, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 37)
        )
        # 1x37

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        # Transformam tensorul de grad superior intr-unul de grad 1, liniar
        out = out.view(-1, 40 * 7 * 7)
        out = self.block3(out)
        return out


# CNN Model utilizand placa grafica - RTX2070
model = CNN()
model.to(device)


@torch.no_grad()
def get_all_preds(modelCNN, loader):
    all_preds = torch.tensor([]).cuda()
    # auxLabelsTensor = torch.tensor([[[]]]).cuda()
    for batch in loader:
        images, auxLabelsTensor = batch
        images = images.permute(0, 3, 1, 2)
        # loadLabelsInThisVariable = torch.cat((loadLabelsInThisVariable, auxLabelsTensor))
        preds = modelCNN(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        ).cuda()
    return all_preds


def getAllLabelsAsTensor(inputDataTrainingTensor):
    labelsTensor = torch.FloatTensor().cuda()
    for batch in inputDataTrainingTensor:
        _, labels1 = batch
        labels1 = labels1.view(-1, 37).type(torch.FloatTensor).cuda()
        labelsTensor = torch.cat((labelsTensor, labels1), dim=0).cuda()
    return labelsTensor


# TODO : Seems to be working but I still think there s something off with this func.
def get_correct_predictions(trained_output, inputLabelTensor):
    correct1 = torch.FloatTensor().cuda()
    trained_output = trained_output.argmax(dim=1)
    correct1 = trained_output.eq(labelTensor).sum()
    return correct1


def create_and_display_confusion_matrix():
    labelTensor = getAllLabelsAsTensor(train_loader1).to(device)
    train_predictions = get_all_preds(model, train_loader1).to(device)

    labelTensor = torch.argmax(labelTensor, dim=1).to(device)
    stackedTensor = torch.stack((labelTensor, train_predictions.argmax(dim=1)), dim=1)
    conf_matrix = torch.zeros(37, 37, dtype=torch.int32)
    # print(conf_matrix)

    for pair in stackedTensor:
        true_label, predicted_label = pair.tolist()
        conf_matrix[true_label, predicted_label] = conf_matrix[true_label, predicted_label] + 1
    print(conf_matrix)

    namesForCMatrix = (
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
        'w',
        'x', 'y', 'z', "space", "number", "period", "comma", "colon", "apostrophe", "hyphen", "semicolon", "question",
        "exclamation", "capitalize")
    plt.figure(figsize=(20, 20))
    plot_cmatrix.plot_confusion_matrix(conf_matrix, namesForCMatrix)


# torch.sum(torch.eq(inputLabelTensor, trained_output))

# Pre-verificare a modelului
print(model)
print("# parameter: ", sum([param.nelement() for param in model.parameters()]))

# Rata de invatare
learning_rate = 0.001

# Using a variable to store the cross entropy method
criterion = nn.CrossEntropyLoss().cuda()

# Using a variable to store the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# list of all train_losses 
train_losses = []

# list of all validation losses 
validation_losses = []

# for loop that iterates over all the epochs
num_epochs = 5
for epoch in range(num_epochs):

    # variables to store/keep track of the loss and number of iterations
    train_loss = 0
    num_iter_train = 0

    # train the model
    model.train()
    for batch in train_loader:
        images, labels = batch
        # Permutam tensorul pentru a corespunde cu inputul modelului [batch_size,channel,width,height]
        images = images.permute(0, 3, 1, 2)
        outputs = model(images)
        # Modelam tensorul ce contine labels, facem squeeze pentru a putea avea dimensiunea asteptata de LossFunc.
        labels = labels.view(-1, 37)
        labelsConverted = torch.argmax(labels, dim=1).cuda()
        # Calculam functia de cost
        loss = criterion(outputs, labelsConverted).cuda()
        # Backward (computes all the gradients)
        optimizer.zero_grad()
        loss.backward()
        # Optimize
        # Loops through all parameters and updates weights by using the gradients
        # Takes steps backwards to optimize (to reach the minimum weight)
        optimizer.step()
        # Update the training loss and number of iterations
        train_loss += loss.data
        num_iter_train += 1

    print('Epoch: {}'.format(epoch + 1))
    print('Training Loss: {:.4f}'.format(train_loss / num_iter_train))
    # append training loss over all the epochs
    train_losses.append(train_loss / num_iter_train)

# Uncomment the following line to plot the confusion matrix corresponding to a training session
create_and_display_confusion_matrix()

# TODO: Do not delete these lines, they account for the evaluation part of the model

# evaluate the model
# model.eval()

# # variables to store/keep track of the loss and number of iterations
# validation_loss = 0
# num_iter_validation = 0
#
# # Iterate over validation_loader
# for i, (images, labels) in enumerate(validation_loader):
#     # need to permute so that the images are of size 3x28x28
#     # essential to be able to feed images into the model
#     images = images.permute(0, 3, 1, 2)
#
#     # Forward, get output
#     outputs = model(images)
#
#     # convert the labels from one hot encoding vectors to integer values
#     labels = labels.view(-1, 37)
#     y_true = torch.argmax(labels, 1).to(device)
#
#     # calculate the validation loss
#     loss = criterion(outputs, y_true).to(device)
#
#     # update the training loss and number of iterations
#     validation_loss += loss.data
#     num_iter_validation += 1
#
# print('Validation Loss: {:.4f}'.format(validation_loss / num_iter_validation))
# # append all validation_losses over all the epochs
# validation_losses.append(validation_loss / num_iter_validation)
#
# num_iter_test = 0
# correct = 0
#
# # Iterate over test_loader
# for images, labels in test_loader:
#     images = images.permute(0, 3, 1, 2)
#
#     # Forward
#     outputs = model(images)
#
#     # convert the labels from one hot encoding vectors into integer values
#     labels = labels.view(-1, 37)
#     y_true = torch.argmax(labels, 1).to(device)
#
#     # find the index of the prediction
#     y_pred = torch.argmax(outputs, 1).type('torch.FloatTensor').to(device)
#
#     # convert to FloatTensor
#     y_true = y_true.type('torch.FloatTensor').to(device)
#
#     # find the mean difference of the comparisons
#     correct += torch.sum(torch.eq(y_true, y_pred).type('torch.FloatTensor')).to(device)
#
# x = (correct / len(test_dataset) * 100)
# print(correct)
# nume = str('{:.4f}'.format(x)) + "model_1_final_highAccuracy.pth"
#
# if x > 99.89:
#     torch.save(model.state_dict(), nume)
#     print('Saved!')
#
# print('Accuracy on the test set: {:.4f}%'.format(correct / len(test_dataset) * 100))
# print()

# learning curve function
# def plot_learning_curve(train_losses, validation_losses):
#     plot the training and validation losses
#     plt.ylabel('Loss')
#     plt.xlabel('Number of Epochs')
#     plt.plot(train_losses, label="training")
#     plt.plot(validation_losses, label="validation")
#     plt.legend(loc=1)
#     plt.show()

# plot the learning curve
# plt.title("Learning Curve (Loss vs Number of Epochs)")
# plot_learning_curve(train_losses, validation_losses)
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]

# torch.save(model.state_dict(), "model_1_final_1.pth")
