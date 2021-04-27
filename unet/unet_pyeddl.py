# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# U-Net implementation using PyEDDL library
#
# References:
# https://arxiv.org/abs/1505.04597
# https://github.com/deephealthproject/pyeddl/blob/master/examples/NN/3_DRIVE/drive_seg.py


import pyeddl.eddl as eddl


def unetPyeddl(layer, use_concat=True):
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

    x4 = eddl.Concat([x4, x5]) if use_concat else eddl.Add([x4, x5])
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.Conv(eddl.UpSampling(x4, [2, 2]), 4*depth, [2, 2], [1, 1], "same")

    x3 = eddl.Concat([x3, x4]) if use_concat else eddl.Add([x3, x4])
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.Conv(eddl.UpSampling(x3, [2, 2]), 2*depth, [2, 2], [1, 1], "same")

    x2 = eddl.Concat([x2, x3]) if use_concat else eddl.Add([x2, x3])
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.Conv(eddl.UpSampling(x2, [2, 2]), depth, [2, 2], [1, 1], "same")

    x = eddl.Concat([x, x2]) if use_concat else eddl.Add([x, x2])
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.Conv(x, 1, [1, 1])

    return x





# Double U-Net construction test
if __name__ == "__main__":
    in_ = eddl.Input((3,256,256))
    out = unetPyeddl(eddl.Input((3,256, 256)))
    net = eddl.Model([in_], [out])
    
    eddl.build(
        net,
        eddl.nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004),
        ["binary_cross_entropy"],
        ["dice"],
        eddl.CS_CPU()
    )

    eddl.summary(net)
