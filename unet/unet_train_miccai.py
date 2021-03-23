# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# U-Net training with the MICCAI 2016 segmentation challenge dataset
# 
# References:
# https://github.com/deephealthproject/use-case-pipelines/blob/3rd_hackathon/python/ms_segmentation_training.py

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from unet          import unet

import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt


MICCAI_PATH = '/home/francisco/Documents/Universidad/5_Carrera/TFG_Computer_Science/datasets/miccai2016/'
TRAIN_SIZE = 5

imgs_path = [MICCAI_PATH + 'preprocessed/s'+str(i)+'/FLAIR_preprocessed.nii' for i in range(1,TRAIN_SIZE)]
masks_path= [MICCAI_PATH + 'unprocessed/s'+str(i)+'/Consensus.nii'           for i in range(1,TRAIN_SIZE)]

# TODO: Out of the whole 144 possilbe slices we only take one, improve in the future!
imgs_np = [np.array(nib.load(img).dataobj)[70,:,:] for img in imgs_path]
msks_np = [np.array(nib.load(msk).dataobj)[70,:,:] for msk in masks_path]
msks_np = [msk/msk.max() for msk in msks_np]

train_imgs = Tensor.fromarray(imgs_np)
train_masks= Tensor.fromarray(msks_np)

net = unet(eddl.Input([1, 512, 512]))

eddl.build(
    net,
    eddl.nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004), 
    ["binary_cross_entropy"],
    ["dice"],
    eddl.CS_CPU(mem="low_mem")
)

eddl.summary(net)
eddl.setlogfile(net, "unet.log")
eddl.plot(net, "unet.pdf")

eddl.fit(net, [train_imgs], [train_masks], 1, 2)
eddl.save(net, "weights_miccai.bin")


