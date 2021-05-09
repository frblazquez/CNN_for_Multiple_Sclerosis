# U-Net for drive dataset with generators
#
# -> Not working in my computer  (batch-size=1)
# -> Working in server cpu (up to batch-size=8)
# -> Not working in server gpu (after getting a CUDA ERROR: Out of memory)

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from memory_profiler import profile


# U-Net implementation
#
# References:
# https://arxiv.org/abs/1505.04597
# https://github.com/zhixuhao/unet/blob/master/model.py
# https://github.com/deephealthproject/pyeddl/blob/master/examples/_OLD/eddl_unet.py
def unet(in_):
    
    # Encoding
    conv1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_  , 64, [3, 3], [1, 1]), True))
    conv1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv1, 64, [3, 3], [1, 1]), True))
    conv2 = eddl.MaxPool(conv1)

    conv2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv2, 128, [3, 3], [1, 1]), True))
    conv2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv2, 128, [3, 3], [1, 1]), True))
    conv3 = eddl.MaxPool(conv2)

    conv3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv3, 256, [3, 3], [1, 1]), True))
    conv3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv3, 256, [3, 3], [1, 1]), True))
    conv4 = eddl.MaxPool(conv3)

    conv4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv4, 512, [3, 3], [1, 1]), True))
    conv4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv4, 512, [3, 3], [1, 1]), True))
    conv4 = eddl.Dropout(conv4, 0.5)
    conv5 = eddl.MaxPool(conv4)

    conv5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv5, 1024, [3, 3], [1, 1]), True))
    conv5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv5, 1024, [3, 3], [1, 1]), True))
    conv5 = eddl.Dropout(conv5, 0.5)
    
    # Decoding
    conv6 = eddl.UpSampling(conv5, [2,2])
    conv6 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv6, 512, [2, 2], [1, 1]), True))
    conv6 = eddl.Concat([conv4, conv6])
    conv6 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv6, 512, [3, 3], [1, 1]), True))
    conv6 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv6, 512, [3, 3], [1, 1]), True))

    conv7 = eddl.UpSampling(conv6, [2,2])
    conv7 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv7, 256, [2, 2], [1, 1]), True))
    conv7 = eddl.Concat([conv3, conv7])
    conv7 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv7, 256, [3, 3], [1, 1]), True))
    conv7 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv7, 256, [3, 3], [1, 1]), True))

    conv8 = eddl.UpSampling(conv7, [2,2])
    conv8 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv8, 128, [2, 2], [1, 1]), True))
    conv8 = eddl.Concat([conv2, conv8])
    conv8 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv8, 128, [3, 3], [1, 1]), True))
    conv8 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv8, 128, [3, 3], [1, 1]), True))

    conv9 = eddl.UpSampling(conv8, [2,2])
    conv9 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv9,  64, [2, 2], [1, 1]), True))
    conv9 = eddl.Concat([conv1, conv9])
    conv9 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv9,  64, [3, 3], [1, 1]), True))
    conv9 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv9,  64, [3, 3], [1, 1]), True))

    # Output
    out = eddl.ReLu(   eddl.BatchNormalization(eddl.Conv(conv9, 2, [3, 3], [1, 1]), True))
    out = eddl.Sigmoid(eddl.BatchNormalization(eddl.Conv(out  , 1, [1, 1], [1, 1]), True))

    return out


def generator(X_data, y_data, batch_size, number_of_batches):
    counter=0
    xbatch = Tensor([batch_size, 3, 512, 512])
    ybatch = Tensor([batch_size, 1, 512, 512])

    while 1:
      eddl.next_batch([X_data, y_data], [xbatch, ybatch])
      counter += 1
      yield xbatch,ybatch
  
      #restart counter to yeild data in the next epoch as well
      if counter >= number_of_batches:
          counter = 0


USE_CONCAT = 1
DRIVE_PATH  = "/scrap/users/blazquez/datasets/drive/"
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")
LOSS_FUNCTION = "mse"
METRICS = "mse"
LEARNING_RATE = 0.00001

@profile
def main(args):
    in_ = eddl.Input([3, 512, 512])
    out = unet(in_)

    # U-Net
    net = eddl.Model([in_],[out])
    eddl.build(
        net,
        eddl.adam(LEARNING_RATE),  # Optimizer
        [LOSS_FUNCTION],           # Losses
        [METRICS],                 # Metrics
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)

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
    gen = generator(x_train, y_train, args.batch_size, args.num_batches)

    print("Starting training")
    for i in range(args.epochs):
        print("\nEpoch %d/%d" % (i + 1, args.epochs))
        eddl.reset_loss(net)
        for j in range(args.num_batches):
            xbatch, ybatch = next(gen)
            eddl.train_batch(net, [xbatch], [ybatch])
            eddl.print_loss(net, j)
            if i == args.epochs - 1:
                yout = eddl.getOutput(out).select(["0"])
                yout.save("./out_%d.jpg" % j)
            print()
    print("All done")   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
