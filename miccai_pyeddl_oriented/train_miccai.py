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
from double_unet import double_unet


MICCAI_PATH  = "/scrap/users/blazquez/datasets/miccai2016/"
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")
LOSS_FUNCTION = "binary_cross_entropy"
METRIC = "mse"
LEARNING_RATE = 0.0001
PROFILE_CMD = 'nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate --format=csv,nounits --id=1 -lms 1000 -f ./profiles/GPUprofile_t4' 

def main(args):
    in_ = eddl.Input([1, 256, 256])
    out = double_unet(in_)
    net = eddl.Model([in_],[out])

    eddl.build(
        net,
        eddl.adam(LEARNING_RATE),  # Optimizer
        [LOSS_FUNCTION],           # Losses
        [METRIC],                  # Metrics
        eddl.CS_GPU(g=[0,1], mem=args.mem)
    )

    #eddl.load(net, "double_unet_miccai.bin")
    eddl.summary(net)
    eddl.setlogfile(net, "run/double_unet_t4.log")

    x_train = Tensor.load(MICCAI_PATH+"bin/miccai_trX_oriented.bin")
    x_train.info()
    y_train = Tensor.load(MICCAI_PATH+"bin/miccai_trY_oriented.bin")
    y_train.info()

    x_batch = Tensor([args.batch_size, 1, 256, 256])
    y_batch = Tensor([args.batch_size, 1, 256, 256])

    x_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsX_oriented.bin")
    x_test.info()
    y_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsY_oriented.bin")
    y_test.info()

    for i in range(args.epochs):
        print("\nEpoch %d/%d\n" % (i + 1, args.epochs))
        eddl.reset_loss(net)
        os.system(PROFILE_CMD + str(i) + '.csv &')
        for j in range(args.num_batches):
            eddl.next_batch([x_train, y_train], [x_batch, y_batch])
            eddl.train_batch(net, [x_batch], [y_batch])
            eddl.print_loss(net, j)
            print()
        os.system('kill $(ps aux | grep nvidia-smi | awk \'{print $2}\')')

        eddl.save(net, "models/double_unet_miccai_"+str(i)+".bin")
        eddl.evaluate(net, [x_test], [y_test])  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=128)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=384)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
