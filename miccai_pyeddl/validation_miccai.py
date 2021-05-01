# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# MICCAI 2016 dataset models validation for Double U-Net

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from double_unet import double_unet


MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"


def main():
    in_ = eddl.Input((1,256,256))
    out = double_unet(in_)
    net = eddl.Model([in_],[out])

    eddl.build(
        net,
        eddl.adam(0.00001),  # Optimizer
        ["binary_cross_entropy"],           # Losses
        ["dice"],                 # Metrics
        eddl.CS_GPU(g=[1,0],mem="low_mem")
    )

    x_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsX_preprocessed.bin")
    y_test = Tensor.load(MICCAI_PATH+"bin/miccai_tsY_preprocessed.bin")

    for i in range(68,80):
        print("Evaluation of the model after epoch "+str(i)+": ")        
        eddl.load(net, "models/double_unet_miccai_"+str(i)+".bin")        
        eddl.evaluate(net, [x_test], [y_test])
        print()


if __name__ == "__main__":
    if MICCAI_PATH == None:
        print("ERROR - Path to DRIVE dataset folder is not set")
    else:
        order = input("WARNING - The evaluation of the several models created can take long. Proceed? (y/n)")
        if order == "y":
            main()
        else:
            print("INFO - Execution aborted")

