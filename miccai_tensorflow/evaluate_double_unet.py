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
model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=[dice])
model.summary()

train_x = np.load(MICCAI_PATH+"npy/miccai2016_trx.npy") 
train_y = np.load(MICCAI_PATH+"npy/miccai2016_try.npy")

test_x = np.load(MICCAI_PATH+"npy/miccai2016_tsx.npy")
test_y = np.load(MICCAI_PATH+"npy/miccai2016_tsy.npy")

dice_test = []
dice_train = []

for i in range(1,129):
    print("INFO - Loading model " + "./unet/models/double_unet_miccai_"+str(i).rjust(4,'0')+".ckpt")
    model.load_weights("./unet/models/unet_miccai_"+str(i).rjust(4,'0')+".ckpt")
  
    scores_test  = model.evaluate(test_x, test_y, batch_size=8, verbose=0)
    scores_train = model.evaluate(train_x, train_y, batch_size=8, verbose=0)

    print(scores_test)
    print(scores_train)

    dice_test.append( scores_test[ 1])
    dice_train.append(scores_train[1])


print("Test dice metrics:")
print(dice_test)

print("Train dice metrics:")
print(dice_train)

    
print("INFO - Execution finished succesfully")

