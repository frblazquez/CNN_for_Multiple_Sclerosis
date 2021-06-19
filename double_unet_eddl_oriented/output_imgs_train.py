import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from double_unet import double_unet


MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"
MODEL_PATH  = "/scrap/users/blazquez/train_oriented/models/double_unet_miccai_"
IMAGES_IDXS = [175, 431, 687]


print("INFO - Building Dobule U-Net")
in_ = eddl.Input([1, 256, 256])
out = double_unet(in_)
net = eddl.Model([in_],[out])

eddl.build(
    net,
    eddl.adam(0.00001),
    ["binary_cross_entropy"],
    ["mse"],
    eddl.CS_GPU(g=[0,1], mem="low_mem") # Much faster with GPU if available
)

print("INFO - Loading test data")
ts_x = Tensor.load(MICCAI_PATH + "bin/miccai_tsX_oriented.bin")
ts_y = Tensor.load(MICCAI_PATH + "bin/miccai_tsY_oriented.bin")

#print("INFO - Saving images and masks")
#for img_idx in IMAGES_IDXS:
#    img_tensor = ts_x.select([str(img_idx)])
#    msk_tensor = ts_y.select([str(img_idx)])
#    msk_tensor.mult_(255.0)
#    img_tensor.save("imgs/ts"+str(img_idx)+"_input.jpg")
#    msk_tensor.save("imgs/ts"+str(img_idx)+"_mask.jpg")


for model_idx in range(104,128):
    print("INFO - Loading model "+str(model_idx))
    eddl.load(net, MODEL_PATH+str(model_idx)+".bin")

    for img_idx in IMAGES_IDXS:
        print("INFO - Saving prediction for image "+str(img_idx))
        img_tensor = ts_x.select([str(img_idx)])
        msk_tensor = ts_y.select([str(img_idx)])
        
        out_tensor = eddl.predict(net, [img_tensor])[0]
        out_tensor.mult_(255.0)
        out_tensor.save("imgs/ts"+str(img_idx)+"_e"+str(model_idx)+"_pred.jpg")

print("INFO - Finished")


