from __future__ import division, print_function, absolute_import

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tflearn
import numpy as np
import cv2
import Image
import csv
from os import listdir
from tifffile import imread
from math import ceil, log

class_map = {0:'building', 1:'barren_land',2:'trees',3:'grassland',4:'road',5:'water'}

tflearn.config.init_graph (num_cores=4, gpu_memory_fraction=0.3)

# Residual blocks
# 32 layers: n=5, 28 layers: n=9, 110 layers: n=18
n = 5

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center([112.3404207,114.68074592,114.19830272], per_channel=True)


# Building Residual Network
net = tflearn.input_data(shape=[None, 28, 28, 3], data_preprocessing=img_prep)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, 6, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='../../checkpoints/resnet_sat6',
                    max_checkpoints=1, tensorboard_verbose=3)

"""
model.fit(X, y, n_epoch=200, validation_set=(X_test, y_test),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=228, shuffle=True,
          run_id='resnet_sat6')
"""

model.load('../../checkpoints/resnet_sat6-4500')

"""
r = csv.reader(open('../utilities/Chandigarh_map.txt', 'r'))
w = csv.writer(open('../../outputs/Chandigarh_patch_classes.csv','w'))

in_path = listdir('../../datasets/Chandigarh_imgs/')

those_that_have_lived = {}

for img_name in in_path:
    those_that_have_lived[img_name] = 1

for row in r:
    if row[0] in those_that_have_lived:
        img = Image.open('../../datasets/Chandigarh_imgs/' + row[0])
        img = img.resize((28,28), Image.ANTIALIAS)
        img = np.asarray(img, dtype=np.float32)
        img /= 255.

        class_this = np.argmax(model.predict([img]))

        w.writerow([class_map[class_this]] + row[1:])
"""

images = listdir('../../datasets/DSTL/three_band_pngs/')

alpha = 0.5

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

for image in ['6050_4_4.png']:
    img = cv2.imread('../../datasets/DSTL/three_band_pngs/' + image)
    img = cv2.resize(img, (int(ceil(img.shape[0]/7)*7), int(ceil(img.shape[1]/7)*7)))
    org_img = img.copy()
    img = img.astype(np.float32)

    for i in range(0,img.shape[0],7):
        for j in range(0,img.shape[1],7):   
            class_this = np.argmax(model.predict([cv2.resize(img[i:i+7,j:j+7],(28,28))]))
            cv2.rectangle(org_img, (j, i), (j+7, i+7),map_color[class_this], -1)
            #print(class_this)


    cv2.imwrite('../../outputs/DSTL/' + image[:-4] + '.png', org_img.astype(np.uint8))
