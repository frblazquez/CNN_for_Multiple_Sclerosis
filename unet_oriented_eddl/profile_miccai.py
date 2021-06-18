# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Double U-Net train over MICCAI 2016 Multiple Sclerosis lessions dataset.

import os
import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from unet import unet
from unet_pyeddl import unetPyeddl

MICCAI_PATH  = "/scrap/users/blazquez/datasets/miccai2016/"
MEM_CHOICE = "full_mem"
LOSS_FUNCTION = "binary_cross_entropy"
METRIC = "mse"
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
PROFILE_CMD = 'nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate --format=csv,nounits --id=0 -lms 1000 -f ./profiles/GPUprofile_t4_' 

def main():
    in_ = eddl.Input([1, 256, 256])
    out = unet(in_)
    net = eddl.Model([in_],[out])

    eddl.build(
        net,
        eddl.adam(LEARNING_RATE),  # Optimizer
        [LOSS_FUNCTION],           # Losses
        [METRIC],                  # Metrics
        eddl.CS_GPU(mem=MEM_CHOICE)
    )
    
    print("INFO - Loading training data")
    x_train = Tensor.load(MICCAI_PATH+"bin/miccai_trX_oriented.bin")
    y_train = Tensor.load(MICCAI_PATH+"bin/miccai_trY_oriented.bin")

    print("INFO - Loading test data")
    x_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsX_oriented.bin")
    y_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsY_oriented.bin")


    x_batch = Tensor([BATCH_SIZE, 1, 256, 256])
    y_batch = Tensor([BATCH_SIZE, 1, 256, 256])


    print("INFO - Profiling training")
    os.system(PROFILE_CMD + 'tr.csv &')
    for j in range(384):
        eddl.next_batch([x_train, y_train], [x_batch, y_batch])
        eddl.train_batch(net, [x_batch], [y_batch])
    os.system('kill $(ps aux | grep nvidia-smi | awk \'{print $2}\')')

    print("INFO - Profiling evaluation")
    os.system(PROFILE_CMD + 'ts.csv &')
    eddl.evaluate(net, [x_test], [y_test], bs=8)  
    os.system('kill $(ps aux | grep nvidia-smi | awk \'{print $2}\')')




if __name__ == "__main__":
    main()
