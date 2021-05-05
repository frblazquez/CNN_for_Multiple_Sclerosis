# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# Double U-Net train with MICCAI2016 dataset

import numpy as np
import tensorflow as tf
import double_unet_keras as duk


MICCAI_PATH = '/home/francisco/Documents/Universidad/5_Carrera/TFG_Computer_Science/miccai2016/'


model = duk.build_model((256, 256, 1))
model.compile(
    optimizer='nadam',
    loss     ='binary_crossentropy',
    metrics  =['mse']
)

train_x = np.load(MICCAI_PATH+"npy/miccai2016_trx.npy") 
train_y = np.load(MICCAI_PATH+"npy/miccai2016_try.npy")

model.fit(train_x, train_y, epochs=5)


print("INFO - Finished")
