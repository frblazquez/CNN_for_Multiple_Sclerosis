# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Double U-Net train over MICCAI 2016 Multiple Sclerosis lessions dataset.

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from double_unet import double_unet


MICCAI_PATH  = "/scrap/users/blazquez/datasets/miccai2016/"
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")
LOSS_FUNCTION = "binary_cross_entropy"
METRICS = "dice"
LEARNING_RATE = 0.0001


def main(args):
    in_ = eddl.Input([1, 256, 256])
    out = double_unet(in_)
    net = eddl.Model([in_],[out])

    eddl.build(
        net,
        eddl.adam(LEARNING_RATE),  # Optimizer
        [LOSS_FUNCTION],           # Losses
        [METRICS],                 # Metrics
        eddl.CS_GPU(g=[0,1],mem=args.mem)
    )

    eddl.summary(net)
    eddl.setlogfile(net, "run/double_unet.log")

    print("Loading train images")
    x_train = Tensor.load(MICCAI_PATH+"bin/miccai_trX_preprocessed.bin")
    x_train.div_(255.0)
    x_train.info()

    print("Loading train masks")
    y_train = Tensor.load(MICCAI_PATH+"bin/miccai_trY_preprocessed.bin")
    y_train.info()

    print("Creating batch tensors")
    x_batch = Tensor([args.batch_size, 1, 256, 256])
    y_batch = Tensor([args.batch_size, 1, 256, 256])

    for i in range(args.epochs):
        print("Epoch %d/%d" % (i + 1, args.epochs))
        eddl.reset_loss(net)
        for j in range(args.num_batches):
            eddl.next_batch([x_train, y_train], [x_batch, y_batch])
            eddl.train_batch(net, [x_batch], [y_batch])
            eddl.print_loss(net, j)
            print()
        print()
        
    eddl.save("models/double_unet_miccai.bin")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=5)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
