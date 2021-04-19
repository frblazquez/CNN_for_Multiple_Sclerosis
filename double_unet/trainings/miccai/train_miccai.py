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
LEARNING_RATE = 0.00001


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

    eddl.load(net, "double_unet_miccai.bin")
    eddl.summary(net)
    eddl.setlogfile(net, "run/double_unet.log")

    x_train = Tensor.load(MICCAI_PATH+"bin/miccai_trX_preprocessed.bin")
    x_train.div_(255.0)
    x_train.info()

    y_train = Tensor.load(MICCAI_PATH+"bin/miccai_trY_preprocessed.bin")
    y_train.info()

    x_batch = Tensor([args.batch_size, 1, 256, 256])
    y_batch = Tensor([args.batch_size, 1, 256, 256])

    x_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsX_preprocessed.bin")
    x_test.div_(255.0)
    x_test.info()

    y_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsY_preprocessed.bin")
    y_test.info()

    for i in range(args.epochs):
        print("Epoch %d/%d" % (i + 1, args.epochs))
        eddl.reset_loss(net)
        for j in range(args.num_batches):
            eddl.next_batch([x_train, y_train], [x_batch, y_batch])
            eddl.train_batch(net, [x_batch], [y_batch])
            eddl.print_loss(net, j)
            print()
        print()
        eddl.save("models/double_unet_miccai_"+str(i)+".bin")
        eddl.evaluate(net, [x_test], [y_test])  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=64)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=1152)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
