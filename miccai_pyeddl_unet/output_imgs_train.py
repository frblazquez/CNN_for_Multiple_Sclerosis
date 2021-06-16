import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from unet import unet


MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"
MODEL_PATH  = "/scrap/users/blazquez/stage2/train_unet/models/unet_miccai_"
IMAGES_IDXS = [175, 943, 1426]


print("INFO - Building Dobule U-Net")
in_ = eddl.Input([1, 256, 256])
out = unet(in_)
net = eddl.Model([in_],[out])

eddl.build(
    net,
    eddl.adam(0.00001),
    ["binary_cross_entropy"],
    ["mse"],
    eddl.CS_GPU(g=[1], mem="low_mem") # Much faster with GPU if available
)

print("INFO - Loading test data")
tr_x = Tensor.load(MICCAI_PATH + "bin/miccai_trX_oriented.bin")
tr_y = Tensor.load(MICCAI_PATH + "bin/miccai_trY_oriented.bin")

#print("INFO - Saving images and masks")
for img_idx in IMAGES_IDXS:
    img_tensor = tr_x.select([str(img_idx)])
    msk_tensor = tr_y.select([str(img_idx)])
    msk_tensor.mult_(255.0)
    img_tensor.save("imgs/tr"+str(img_idx)+"_input.jpg")
    msk_tensor.save("imgs/tr"+str(img_idx)+"_mask.jpg")


for model_idx in range(128):
    print("INFO - Loading model "+str(model_idx))
    eddl.load(net, MODEL_PATH+str(model_idx)+".bin")

    for img_idx in IMAGES_IDXS:
        print("INFO - Saving prediction for image "+str(img_idx))
        img_tensor = tr_x.select([str(img_idx)])
        msk_tensor = tr_y.select([str(img_idx)])
        
        out_tensor = eddl.predict(net, [img_tensor])[0]
        out_tensor.mult_(255.0)
        out_tensor.save("imgs/tr"+str(img_idx)+"_e"+str(model_idx)+"_pred.jpg")

print("INFO - Finished")


