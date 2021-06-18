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
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")
LOSS_FUNCTION = "binary_cross_entropy"
METRIC = "mse"
LEARNING_RATE = 0.0001
PROFILE_CMD = 'nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate --format=csv,nounits --id=1 -lms 1000 -f ./profiles/GPUprofile_t4_' 

def main():
    in_ = eddl.Input([1, 256, 256])
    out = unet(in_)
    net = eddl.Model([in_],[out])

    eddl.build(
        net,
        eddl.adam(LEARNING_RATE),  # Optimizer
        [LOSS_FUNCTION],           # Losses
        [METRIC],                  # Metrics
        eddl.CS_GPU(g=[0,1], mem="low_mem")
    )
    
    eddl.summary(net)
    eddl.plot(net, "unet_plot.pdf")

    print("INFO - Execution finished")


if __name__=="__main__":
    main()
    
