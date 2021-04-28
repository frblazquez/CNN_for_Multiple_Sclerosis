# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# U-Net model train over Drive dataset
#
# References:
# https://drive.grand-challenge.org/
# https://github.com/deephealthproject/pyeddl/blob/master/examples/NN/3_DRIVE/drive_seg.py

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from unet import unet


LOSS_FUNCTION = "mse"
METRICS       = "mse"
LEARNING_RATE = 0.00001
DRIVE_PATH  = None
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):
    # unet for image segmentation
    print("Creating model structure")
    net = unet(eddl.Input([3, 512, 512]))

    print("Building model")
    eddl.build(
        net,
        eddl.adam(0.00001), # Optimizer
        [LOSS_FUNCTION],    # Losses
        [METRICS],          # Metrics
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )

    print("Model summary:")
    eddl.summary(net)
    eddl.setlogfile(net, "models/unet_drive.log")

    print("Reading training data")
    x_train = Tensor.load(DRIVE_PATH+"drive_trX_preprocessed.bin")
    x_train.div_(255.0)
    x_train.info()

    print("Reading test data")
    y_train = Tensor.load(DRIVE_PATH+"drive_trY_preprocessed.bin")
    y_train.div_(255.0)
    y_train.info()

    xbatch = Tensor([args.batch_size, 3, 512, 512])
    ybatch = Tensor([args.batch_size, 1, 512, 512])

    print("Starting training")
    for i in range(args.epochs):
        print("\nEpoch %d/%d" % (i + 1, args.epochs))
        eddl.reset_loss(net)
        for j in range(args.num_batches):
            eddl.next_batch([x_train, y_train], [xbatch, ybatch])
            eddl.train_batch(net, [xbatch_da], [ybatch_da])
            eddl.print_loss(net, j)
            #if i == args.epochs - 1:
            #    yout = eddl.getOutput(out).select(["0"])
            #    yout.save("./out_%d.jpg" % j)
            #print()

    print("Training successfully done, saving model...")
    eddl.save(net, "models/unet_drive.bin")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=50)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
