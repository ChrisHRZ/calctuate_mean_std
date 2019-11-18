import numpy as np
import cv2
import random
import os
import tqdm
import torch
import torchvision.transforms as tf
 
# calculate means and std

i = 0

img_h, img_w = 512, 512
# imgs = np.zeros([img_w, img_h, 3, 1])
# imgs = torch.from_numpy(imgs)
# imgs = imgs.cuda()
means, stdevs = [], []

files = os.listdir(img_path)
files.sort()
 
for f in files:
    print(f)
    if not f.endswith('.png'):
        continue
    img = cv2.imread(img_path+f)
    img = cv2.resize(img, (img_h, img_w))
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    img = img.cuda()
    if i==0:
        imgs = img
    else:
        imgs = torch.cat((imgs, img), 0)
    i+=1
#         print(i)
 
imgs = imgs/255.
 
 
for i in tqdm_notebook(range(3)):
    pixels = imgs[:,:,:,i]
    means.append(torch.mean(pixels))
    stdevs.append(torch.sqrt(torch.var(pixels)))
 
# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse() # BGR --> RGB
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
