import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt

def GramMatrixCalcu(input):
    a, b, c, d = input.shape
    G = np.zeros((b,b))
    for i in range(a):
        feature = input[i,:].reshape(b, c*d)
        G += np.dot(feature, feature.T)
    G /= a
    return G

img_transform = T.Compose([T.ToTensor()])
#test_img_transform = T.Compose([T.CenterCrop(256), T.Resize(256), T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

"""
data = ImageFolder(root='../Data/PrjData', transform=img_transform)
data1 = ImageFolder(root='./StarGAN/data/RaFD/train/', transform=img_transform)
data_loader = torch.utils.data.DataLoader(dataset=data1,batch_size=1,shuffle=False, num_workers=0)
count_0=0
count_1=0
count_2=0
for idx, data in enumerate(data_loader):
    x_fixed, c_org = data
    if np.asscalar(c_org.numpy()) == 0:
        count_0 += 1
    elif np.asscalar(c_org.numpy()) == 1:
        count_1 += 1
    elif np.asscalar(c_org.numpy()) == 2:
        count_2 += 1
print(count_0, count_1, count_2)

print(x_fixed, c_org)
"""
"""
Real_imgs = ImageFolder(root='../data/RaFD/test', transform=test_img_transform)
all_real_imgs = {}
test_names = ['ChineseInk', 'Morandi', 'Nature', 'OldPhotos', 'Picasso', 'Raphael', 'VanGogh' ]
for name in test_names:
    all_real_imgs[name] = []

for i in range(len(Real_imgs)):
    all_real_imgs[test_names[int(Real_imgs[i][1])]].append(Real_imgs[i][0].numpy().tolist())

for k, v in all_real_imgs.items():
    all_real_imgs[k] = np.array(v)

for k, v in all_real_imgs.items():
    print(k, 'shape', v.shape)
    print(GramMatrixCalcu(v))
"""
Result_imgs = ImageFolder(root='test_image_all/all', transform=img_transform)
all_results_imgs = {}
folder_names = ['1-original', '2-ink', '3-morandi', '4-nature', '5-oldphoto', '6-picasso', '7-raphael', '8-vangogh', '9-37', '10-24', '11-28', '12-58', '13-25' ]
for name in folder_names:
    all_results_imgs[name] = []

for i in range(len(Result_imgs)):
    all_results_imgs[folder_names[int(Result_imgs[i][1])]].append(Result_imgs[i][0].numpy().tolist())

for k, v in all_results_imgs.items():
    all_results_imgs[k] = np.array(v)

for k, v in all_results_imgs.items():
    print(k, 'shape', v.shape)
    print(GramMatrixCalcu(v))

"""
print(GramMatrixCalcu(ink_imgs[0].reshape(-1, 3, 256, 256)))
print(GramMatrixCalcu(ink_imgs[50].reshape(-1, 3, 256, 256)))
print(GramMatrixCalcu(ink_imgs[0:40]))
print(GramMatrixCalcu(ink_imgs[0:150]))
"""
#print(GramMatrixCalcu(ink_imgs))

Result_imgs = ImageFolder(root='test_image_NoC_all/all', transform=img_transform)
all_results_imgs = {}
folder_names = ['1-original', '2-ink', '3-morandi', '4-nature', '5-oldphoto', '6-picasso', '7-raphael', '8-vangogh', '9-37', '10-24', '11-28', '12-58', '13-25' ]
for name in folder_names:
    all_results_imgs[name] = []

for i in range(len(Result_imgs)):
    all_results_imgs[folder_names[int(Result_imgs[i][1])]].append(Result_imgs[i][0].numpy().tolist())

for k, v in all_results_imgs.items():
    all_results_imgs[k] = np.array(v)

for k, v in all_results_imgs.items():
    print(k, 'shape', v.shape)
    print(GramMatrixCalcu(v))

