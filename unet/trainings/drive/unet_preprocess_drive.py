# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Drive dataset preprocessing for U-Net model
#
# References:
# https://drive.grand-challenge.org/
# https://github.com/deephealthproject/pyeddl/blob/master/examples/NN/3_DRIVE/drive_seg.py

import os
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

# Set the following variable to the folder containing the DRIVE dataset
DRIVE_PATH = None

def main():
    # Uncomment to download the DRIVE dataset
    # eddl.download_drive()

    # Get the image and mask and scale and crop (to 512x512) them both
    in_1  = eddl.Input([3, 584, 584])
    in_2  = eddl.Input([1, 584, 584])
    layer = eddl.Concat([in_1, in_2])

    layer = eddl.RandomCropScale(layer, [0.9, 1.0])
    layer = eddl.CenteredCrop(layer, [512, 512])
    img   = eddl.Select(layer, ["0:3"])
    mask  = eddl.Select(layer, ["3"])

    danet = eddl.Model([in_1, in_2], [])
    eddl.build(danet,cs=eddl.CS_CPU())
    eddl.summary(danet)

    drive_imgs = Tensor.load(DRIVE_PATH+"drive_trX.bin")
    drive_msks = Tensor.load(DRIVE_PATH+"drive_trY.bin")

    eddl.forward(danet, [drive_imgs, drive_msks])
    drive_imgs_processed = eddl.getOutput(img)
    drive_msks_processed = eddl.getOutput(mask)

    drive_imgs_processed.save("drive_trX_preprocessed.bin")
    drive_msks_processed.save("drive_trY_preprocessed.bin")

    os.rename("drive_trX_preprocessed.bin", DRIVE_PATH+"drive_trX_preprocessed.bin")
    os.rename("drive_trY_preprocessed.bin", DRIVE_PATH+"drive_trY_preprocessed.bin")


# TODO: Test!
if __name__ == "__main__":
    if DRIVE_PATH == None:
        print("ERROR - Path to DRIVE dataset folder is not set")
    else
        order = input("WARNING - Data preprocessing must be executed only once. Proceed? (y/n)")
        if order == "y":
            main()
        else
            print("INFO - Execution aborted")

            

