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
import time

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from double_unet import double_unet
from unet_pyeddl import unetPyeddl
from unet import unet

MICCAI_PATH  = "/scrap/users/blazquez/datasets/miccai2016/"
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")
LOSS_FUNCTION = "binary_cross_entropy"
METRIC = "mse"
LEARNING_RATE = 0.0001
PROFILE_CMD = 'nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate --format=csv,nounits --id=0 -lms 500 -f ./profiles/GPUprofile_unet_t4_' 
TRAIN_OR_EVALUATE=False

def main(args):
    in_ = eddl.Input([1, 256, 256])
    out = unet(in_)
    net = eddl.Model([in_],[out])

    eddl.build(
        net,
        eddl.adam(LEARNING_RATE),  # Optimizer
        [LOSS_FUNCTION],           # Losses
        [METRIC],                  # Metrics
        eddl.CS_CPU(mem=args.mem)
    )
    
    #eddl.load(net,"unet_miccai.bin")
    #eddl.summary(net)
    #eddl.setlogfile(net, "run/unet_t4.log")

    x_train = Tensor.load(MICCAI_PATH+"bin/miccai_trX_oriented.bin")
    y_train = Tensor.load(MICCAI_PATH+"bin/miccai_trY_oriented.bin")

    x_batch = Tensor([args.batch_size, 1, 256, 256])
    y_batch = Tensor([args.batch_size, 1, 256, 256])

    x_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsX_oriented.bin")
    y_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsY_oriented.bin")

    for i in range(1):
        print("\nEpoch %d/%d\n" % (i + 1, args.epochs))
        eddl.reset_loss(net)

        if TRAIN_OR_EVALUATE:
            start_time = time.time()
            for j in range(args.num_batches):
                start_batch = time.time()
                eddl.next_batch([x_train, y_train], [x_batch, y_batch])
                eddl.train_batch(net, [x_batch], [y_batch])
                end_batch = time.time()
            
                #eddl.print_loss(net, j)
                #print()

                print("Batch time: {0}".format(end_batch - start_batch))
        #end_time = time.time()
        #print("Training epoch time: {0}".format(end_time - start_time))
        #eddl.save(net, "models/unet_miccai_"+str(i)+".bin")
        
        else:
            start_time = time.time()
            eddl.evaluate(net, [x_test], [y_test])  
            end_time = time.time()
            print("Validation data evaluation time: ".format(end_time - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=128)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=384)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
