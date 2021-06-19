# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# U-Net training with tensorflow over MICCAI2016 dataset.

import os
import numpy as np
import matplotlib.pyplot as plt

from unet_keras import unet
from double_unet_keras import build_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import keras.backend as K


MICCAI_PATH = '/scrap/users/blazquez/datasets/miccai2016/'


def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

in_ = Input((256, 256, 1))
out = unet(in_)

model = Model(in_, out)
model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['mean_squared_error', 'binary_accuracy', dice])
model.summary()

train_x = np.load(MICCAI_PATH+"npy/miccai2016_trx.npy") 
train_y = np.load(MICCAI_PATH+"npy/miccai2016_try.npy")

test_x = np.load(MICCAI_PATH+"npy/miccai2016_tsx.npy")
test_y = np.load(MICCAI_PATH+"npy/miccai2016_tsy.npy")

train_idxs = [175, 943, 1426]
test_idxs  = [175, 431,  687] 

for i in range(1,129):
    print("INFO - Loading model " + "./unet/models/double_unet_miccai_"+str(i).rjust(4,'0')+".ckpt")
    model.load_weights("./unet/models/unet_miccai_"+str(i).rjust(4,'0')+".ckpt")

    train_imgs_pred = 255*model.predict(train_x[train_idxs])
    test_imgs_pred  = 255*model.predict(test_x[test_idxs])

    for j in [0,1,2]:
        plt.imsave("imgs/tr"+str(train_idxs[j])+"/tr"+str(train_idxs[j])+"_e"+str(i)+"_pred.jpg", np.squeeze(train_imgs_pred[j]), cmap="gray")
       
    for j in [0,1,2]:
        plt.imsave("imgs/ts"+str(test_idxs[j]) +"/ts"+str(test_idxs[j]) +"_e"+str(i)+"_pred.jpg", np.squeeze( test_imgs_pred[j]), cmap="gray")
  
    #model.evaluate(test_x, test_y, batch_size=8)
    #model.evaluate(train_x, train_y, batch_size=8)


    
print("INFO - Execution finished succesfully")

