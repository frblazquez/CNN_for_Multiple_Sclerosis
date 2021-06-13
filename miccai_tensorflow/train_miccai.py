# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# Double U-Net train with MICCAI2016 dataset

import numpy as np
import tensorflow as tf
from double_unet_keras import build_model


MICCAI_PATH = '/scrap/users/blazquez/datasets/miccai2016/'


model = build_model((256, 256, 1))
model.compile(
    optimizer='nadam',
    loss     ='binary_crossentropy',
    metrics  =['mse']
)

train_x = np.load(MICCAI_PATH+"npy/miccai2016_trx.npy") 
train_y = np.load(MICCAI_PATH+"npy/miccai2016_try.npy")

model.fit(train_x, train_y, epochs=5)


print("INFO - Finished")
