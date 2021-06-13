# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# U-Net training with tensorflow over MICCAI2016 dataset.

import os
import numpy as np
import time

from unet_keras import unet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint


MICCAI_PATH = '/scrap/users/blazquez/datasets/miccai2016/'


class EvaluateAfterEachEpoch(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        scores = self.model.evaluate(x, y, verbose=False)
        print('\nTesting loss: {}, mse: {}, bin_accuracy: {}\n'.format(scores[0], scores[1], scores[2]))


in_ = Input((256, 256, 1))
out = unet(in_)

model = Model(in_, out)
model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['mean_squared_error', 'binary_accuracy'])
model.summary()

train_x = np.load(MICCAI_PATH+"npy/miccai2016_trx.npy") 
train_y = np.load(MICCAI_PATH+"npy/miccai2016_try.npy")

test_x = np.load(MICCAI_PATH+"npy/miccai2016_tsx.npy")
test_y = np.load(MICCAI_PATH+"npy/miccai2016_tsy.npy")

#cp_callback = ModelCheckpoint(filepath="models/double_unet_miccai_{epoch:04d}.ckpt", save_weights_only=True, verbose=1)
#ev_callback = EvaluateAfterEachEpoch((test_x, test_y))

start_time = time.time()
os.system('nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate, --format=csv,nounits --id=0 -lms 500 -f ./profiles/GPUprofile_unet_t4_tr.csv &') 
model.fit(train_x, train_y, epochs=1, batch_size=8) #callbacks=[cp_callback, ev_callback])
os.system('kill $(ps aux | grep \'nvidia-smi\' | awk \'{print $2}\')')
end_time = time.time()

print("Epoch training time: {0}".format((end_time-start_time)))


start_time = time.time()
os.system('nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate, --format=csv,nounits --id=0 -lms 500 -f ./profiles/GPUprofile_unet_t4_ts.csv &')
model.evaluate(test_x,  test_y,  batch_size=8) #callbacks=[cp_callback, ev_callback])
model.evaluate(train_x, train_y, batch_size=8) #callbacks=[cp_callback, ev_callback])
os.system('kill $(ps aux | grep \'nvidia-smi\' | awk \'{print $2}\')')
end_time = time.time()

print("Evaluation time: {0}".format((end_time-start_time)/5))


print("INFO - Execution finished succesfully")

