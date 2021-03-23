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

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from unet          import unet


ISBI_PATH  = "/home/francisco/Documents/Universidad/5_Carrera/TFG_Computer_Science/datasets/isbi/train/"
TRAIN_SIZE = 30

imgs_path = [ISBI_PATH + 'image_lq/'+str(i)+'.png' for i in range(TRAIN_SIZE)]
masks_path= [ISBI_PATH + 'label_lq/'+str(i)+'.png' for i in range(TRAIN_SIZE)]

train_imgs = Tensor.fromarray([Tensor.load(img).getdata()          for img in imgs_path])
train_masks= Tensor.fromarray([Tensor.load(msk).div(255).getdata() for msk in masks_path])

net = unet(eddl.Input([1, 256, 256]))

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

eddl.fit(net, [train_imgs], [train_masks], 2, 5)

