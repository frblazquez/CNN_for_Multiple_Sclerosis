# U-Net light for drive dataset with generators
#
# -> Not working in my computer  (batch-size=1)
# -> Working in server cpu (up to batch-size=10)
# -> Working in server gpu (up to batch-size=10)

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


def unetLight(layer):
    x = layer
    depth = 32

    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x2 = eddl.MaxPool(x, [2, 2], [2, 2])
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])
    x5 = eddl.LeakyReLu(eddl.Conv(x5, 8*depth, [3, 3], [1, 1], "same"))
    x5 = eddl.LeakyReLu(eddl.Conv(x5, 8*depth, [3, 3], [1, 1], "same"))
    x5 = eddl.Conv(eddl.UpSampling(x5, [2, 2]), 8*depth, [2, 2], [1, 1], "same")

    x4 = eddl.Concat([x4, x5]) if USE_CONCAT else eddl.Add([x4, x5])
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.Conv(eddl.UpSampling(x4, [2, 2]), 4*depth, [2, 2], [1, 1], "same")

    x3 = eddl.Concat([x3, x4]) if USE_CONCAT else eddl.Add([x3, x4])
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.Conv(eddl.UpSampling(x3, [2, 2]), 2*depth, [2, 2], [1, 1], "same")

    x2 = eddl.Concat([x2, x3]) if USE_CONCAT else eddl.Add([x2, x3])
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.Conv(eddl.UpSampling(x2, [2, 2]), depth, [2, 2], [1, 1], "same")

    x = eddl.Concat([x, x2]) if USE_CONCAT else eddl.Add([x, x2])
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.Conv(x, 1, [1, 1])

    return x


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


def main(args):
    in_ = eddl.Input([3, 512, 512])
    out = unetLight(in_)

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
