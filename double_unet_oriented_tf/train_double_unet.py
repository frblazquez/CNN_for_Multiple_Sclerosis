# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# U-Net training with tensorflow over MICCAI2016 dataset.

import os
import numpy as np

from double_unet_keras import build_model
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
out = build_model(in_)

model = Model(in_, out)
model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['mean_squared_error', 'binary_accuracy'])
model.summary()

train_x = np.load(MICCAI_PATH+"npy/miccai2016_trx.npy") 
train_y = np.load(MICCAI_PATH+"npy/miccai2016_try.npy")

test_x = np.load(MICCAI_PATH+"npy/miccai2016_tsx.npy")
test_y = np.load(MICCAI_PATH+"npy/miccai2016_tsy.npy")

cp_callback = ModelCheckpoint(filepath="models/double_unet_miccai_{epoch:04d}.ckpt", save_weights_only=True, verbose=1)
ev_callback = EvaluateAfterEachEpoch((test_x, test_y))

os.system('nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate, --format=csv,nounits --id=0 -lms 500 -f ./profiles/GPUprofile_v100.csv &') 
model.fit(train_x, train_y, epochs=128, batch_size=8, callbacks=[cp_callback, ev_callback])
os.system('kill $(ps aux | grep \'nvidia-smi\' | awk \'{print $2}\')')

print("INFO - Execution finished succesfully")

