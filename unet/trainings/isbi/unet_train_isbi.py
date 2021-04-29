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


DRIVE_PATH  = "/home/francisco/Documents/Universidad/5_Carrera/TFG_Computer_Science/datasets/drive/"
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):
    # DA net, for image preprocessing
    in_1 = eddl.Input([3, 584, 584])
    in_2 = eddl.Input([1, 584, 584])
    layer = eddl.Concat([in_1, in_2])

    layer = eddl.RandomCropScale(layer, [0.9, 1.0])
    layer = eddl.CenteredCrop(layer, [512, 512])
    img = eddl.Select(layer, ["0:3"])
    mask = eddl.Select(layer, ["3"])

    danet = eddl.Model([in_1, in_2], [])
    eddl.build(danet)
    if args.gpu:
        eddl.toGPU(danet, mem="low_mem")
    eddl.summary(danet)

    # SegNet, for image segmentation
    in_ = eddl.Input([3, 512, 512])
    #out = eddl.Sigmoid(UNetWithPadding(in_))
    #segnet = eddl.Model([in_], [out])
    #segnet = double_unet(in_)
    eddl.build(
        segnet,
        eddl.adam(0.00001),  # Optimizer
        ["mse"],  # Losses
        ["mse"],  # Metrics
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(segnet)

    print("Reading training data")
    # x_train_f = Tensor.fromarray(np.load("drive_trX.npy").astype(np.float32))
    x_train_f = Tensor.load(DRIVE_PATH+"drive_trX.bin")
    x_train = x_train_f.permute([0, 3, 1, 2])
    x_train.info()
    x_train.div_(255.0)

    print("Reading test data")
    # y_train = Tensor.fromarray(np.load("drive_trY.npy").astype(np.float32))
    y_train = Tensor.load(DRIVE_PATH+"drive_trY.bin")
    y_train.info()
    y_train.reshape_([20, 1, 584, 584])
    y_train.div_(255.0)

    xbatch = Tensor([args.batch_size, 3, 584, 584])
    ybatch = Tensor([args.batch_size, 1, 584, 584])

    print("Starting training")
    for i in range(args.epochs):
        print("\nEpoch %d/%d" % (i + 1, args.epochs))
        eddl.reset_loss(segnet)
        for j in range(args.num_batches):
            eddl.next_batch([x_train, y_train], [xbatch, ybatch])
            # DA net
            eddl.forward(danet, [xbatch, ybatch])
            xbatch_da = eddl.getOutput(img)
            ybatch_da = eddl.getOutput(mask)
            # SegNet
            eddl.train_batch(segnet, [xbatch_da], [ybatch_da])
            eddl.print_loss(segnet, j)
            if i == args.epochs - 1:
                yout = eddl.getOutput(out).select(["0"])
                yout.save("./out_%d.jpg" % j)
            print()
    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=50)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
