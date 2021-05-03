# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Double U-Net models evaluation over certain selected images, the network
# output for every model is saved as .jpg image


import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from double_unet import double_unet


MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"
MODEL_PATH  = "/scrap/users/blazquez/train/models/double_unet_miccai_"
IMAGES_IDXS = [1191, 1266, 1273, 1975, 1252,  415, 1634]
NUM_MODELS = 80

print("INFO - Building Dobule U-Net")
in_ = eddl.Input([1, 256, 256])
out = double_unet(in_)
net = eddl.Model([in_],[out])

eddl.build(
    net,
    eddl.adam(0.00001),
    ["binary_cross_entropy"],
    ["mse"],
    eddl.CS_GPU(g=[0,1], mem="low_mem")
)

print("INFO - Loading test data")
ts_x = Tensor.load(MICCAI_PATH + "bin/miccai_tsX_preprocessed.bin")
ts_y = Tensor.load(MICCAI_PATH + "bin/miccai_tsY_preprocessed.bin")
ts_x.div_(255.0)

for model_idx in range(NUM_MODELS):
    print("INFO - Loading model "+str(model_idx))
    eddl.load(net, MODEL_PATH+str(model_idx)+".bin")

    for img_idx in IMAGES_IDXS:
        print("INFO - Saving prediction for image "+str(img_idx)+" of model "+str(model_idx))
        img_tensor = ts_x.select([str(img_idx)])
        msk_tensor = ts_y.select([str(img_idx)])
        
        out_tensor = eddl.predict(net, [img_tensor])[0]
        out_tensor.mult_(255.0)
        out_tensor.save("imgs/ts"+str(img_idx)+"_e"+str(model_idx)+"_pred.jpg")

print("INFO - Finished")
