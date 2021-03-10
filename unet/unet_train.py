# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# U-Net training with the ISBI challenge dataset
# 
# References:
# https://github.com/deephealthproject/use-case-pipelines/blob/3rd_hackathon/python/ms_segmentation_training.py

import argparse
import sys
import pyeddl.eddl as eddl
import unet

from pyeddl.tensor import Tensor


ISBI_PATH   = "/home/francisco/Documents/Universidad/5º_Carrera/TFG_Computer_Science/datasets/isbi/train/"
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):
    size = [256, 256] 
    in_  = eddl.Input([3, size[0], size[1]])
    net  = unet.unet(in_)

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )

    eddl.summary(net)
    eddl.setlogfile(net, "unet.log")
    eddl.plot(net, "unet.pdf")

    #for i in range(25):
    #    x_train = Tensor.load(ISBI_PATH+"image/"+str(i)+".png")
    #    y_train = Tensor.load(ISBI_PATH+"label/"+str(i)+".png")
    #
    #for i in range(25,30):
    #    x_test = Tensor.load(ISBI_PATH+"image/"+str(i)+".png")
    #    y_test = Tensor.load(ISBI_PATH+"label/"+str(i)+".png")


    #x_train = []
    #y_train = []
    #for i in range(25):
    #    x_train.append(Tensor.load(ISBI_PATH+"image/"+str(i)+".png"))
    #    y_train.append(Tensor.load(ISBI_PATH+"label/"+str(i)+".png"))
    #
    #x_test = []
    #y_test = []
    #for i in range(25,30):
    #    x_test.append(Tensor.load(ISBI_PATH+"image/"+str(i)+".png"))
    #    y_test.append(Tensor.load(ISBI_PATH+"label/"+str(i)+".png"))

    training_augs   = ecvl.SequentialAugmentationContainer([ecvl.AugResizeDim(size)])
    validation_augs = ecvl.SequentialAugmentationContainer([ecvl.AugResizeDim(size)])
    dataset_augs    = ecvl.DatasetAugmentations([training_augs, validation_augs, None])

    print('Reading dataset')
    d = ecvl.DLDataset(args.in_ds, args.batch_size, dataset_augs, ecvl.ColorType.none, ecvl.ColorType.none)
    #v = MSVolume(d, args.n_channels)  # MSVolume takes a reference to DLDataset

    # Prepare tensors which store batches
    #x = Tensor([args.batch_size, args.n_channels, size[0], size[1]])
    #y = Tensor([args.batch_size, args.n_channels, size[0], size[1]])

    #for i in range(args.epochs):
    #    eddl.fit(net, [x_train], [y_train], args.batch_size, 1)
    #    eddl.evaluate(net, [x_test], [y_test], bs=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_ds', metavar='INPUT_DATASET')
    parser.add_argument('--epochs', type=int, metavar='INT', default=100)
    parser.add_argument('--batch_size', type=int, metavar='INT', default=16)
    parser.add_argument('--num_classes', type=int, metavar='INT', default=1)
    parser.add_argument('--n_channels', type=int, metavar='INT', default=1, help='Number of slices to stack together and use as input')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--size', type=int, metavar='INT', default=256, help='Size of input slices')
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument('--out-dir', metavar='DIR', help='if set, save images in this directory')
    parser.add_argument('--ckpts', type=str)
    main(parser.parse_args())
