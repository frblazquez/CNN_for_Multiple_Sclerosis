# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:


import numpy as np

from unet_keras import unet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam


MICCAI_PATH = '/home/francisco/Documents/Universidad/5_Carrera/TFG_Computer_Science/miccai2016/'


in_ = Input((256, 256, 1))
out = unet(in_)

model = Model(in_, out)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

train_x = np.load(MICCAI_PATH+"npy/miccai2016_trx.npy") 
train_y = np.load(MICCAI_PATH+"npy/miccai2016_try.npy")

print(train_x.shape)
print(train_y.shape)
