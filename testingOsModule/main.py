from unipath.path import Path
import os
import string
from PIL import Image
import numpy as np
import cv2
import torch

# os.chdir('E:\\the VIDE')
# print(os.getcwd())
# print(os.listdir())

# for target in os.listdir('E:\\the VIDE'):
#     print(target)

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

path = Path(os.getcwd())
print(path.parent)

i = 0
for tgt in os.listdir(path.parent + "/dataset"):
    for folder in os.listdir(path.parent + "/dataset/" + tgt + "/Uploaded"):
        if folder == ".DS_Store":
            os.remove(path.parent + "/dataset/" + tgt + "/Uploaded/" + folder)
            print(folder)
        for filename in os.listdir(path.parent + "/dataset/" + tgt + "/Uploaded/" + folder):
            if filename == ".DS_Store":
                print("FOUND")
                i += 1
                os.remove(path.parent + "/dataset/" + tgt + "/Uploaded/" + folder + "/" + filename)
                i -= 1
                print(i)

for tgt in os.listdir(path.parent + "/dataset"):
    for folder in os.listdir(path.parent + "/dataset/" + tgt + "/Uploaded"):
        for filename in os.listdir(path.parent + "/dataset/" + tgt + "/Uploaded/" + folder):
            # Obtinem imaginea si denumirea
            picture = []
            curr_target = matrice_encoding[tgt]
            image = Image.open(path.parent + "/dataset/" + tgt + "/Uploaded/" + folder + "/" + filename)
            image = image.convert('RGB')
            image = np.array(image)
            # image = image.astype(np.float32) / 255.0
            # h = image.shape[0]
            # w = image.shape[1]
            # image2 = image.tolist()
            # for i in range(0, h):
            #     for j in range(0, w):
            #         for k in range(len(image[i, j])):
            #             pixelValue = str(image[i, j, k])
            #             if (int(pixelValue)) < 159:
            #                 image[i, j, k] = 20
            # image2 = Image.fromarray(image).save('C:\\Users\mariusfacepoze\Desktop\modified_pixels.jpg',
            #                                      'JPEG')
            # print(image)
            # Redimensionam imaginea la 28x28x3
            image = cv2.resize(image, (28, 28))
            # Normalizare 0-1
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            picture.append(image)
            # Convertim linia unei matrici ce corespunde unui caracter intr-un tensor
            curr_target = torch.Tensor([curr_target])
            # Vectorul picture primeste o poza si denumirea acesteia concatenate pe pozitii consecutive ex : [0] & [1]
            picture.append(curr_target)
            data.append(picture)

caractere = alfabet + simboluri

index2char = {}
poz = 0
for caracter in caractere:
    index2char[poz] = caracter
    poz += 1
print(index2char[1])

test_dataset = data[22000:24400]


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


test_chars = num_chars(test_dataset, index2char)
num = 0
for caracter in caractere:
    if caracter in test_chars:
        num += 1
    else:
        break
print(num)
