import os
import numpy as np

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from pyeddl._core  import Metric
from unet import unet

class BinaryAccuracy(Metric):
    def __init__(self, threshold=0.5):
        Metric.__init__(self, "py_binary_accuracy")
        self.threshold = threshold

    def value(self, msk, out):        
        out_np = out.getdata() < self.threshold
        msk_np = msk.getdata().astype(np.bool)
        res_np = np.logical_xor(msk_np, out_np)

        #print("Batch shape: [{0}, {1}, {2}]".format(msk_np.shape[0], msk_np.shape[1], msk_np.shape[2]))
        #print("Batch binary accuracy: {0}".format(np.sum(res_np) / (y_np.shape[1]*y_np.shape[2])))

        return np.sum(res_np) / (msk_np.shape[2]*msk_np.shape[3])


class Dice(Metric):
    def __init__(self, threshold=0.5):
        Metric.__init__(self, "py_dice")
        self.threshold = threshold

    def value(self, msk, out):   
        out_np = out.getdata() >= self.threshold
        msk_np = msk.getdata().astype(np.bool)
        intersection = np.logical_and(msk_np, out_np)
        
        dice = 0
        for i in range(msk_np.shape[0]):
            union_i = np.sum(msk_np[i]) + np.sum(out_np[i])
            if union_i == 0:
                dice += 1
            else:
                dice += 2*np.sum(intersection[i]) / union_i

        #print("Batch shape: [{0}, {1}, {2}]".format(y_np.shape[0], y_np.shape[1], y_np.shape[2]))
        #print("Batch dice: {0}".format(dice)

        return dice



MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"
MODEL_PATH  = "/scrap/users/blazquez/stage2/train_unet/models/unet_miccai_"
PROFILE_CMD = "nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu,pstate --format=csv,nounits --id=0 -lms 1000 -f ./profiles/tr_dice/GPUprofile_t4_tr"

print("INFO - Building Dobule U-Net")
in_ = eddl.Input([1, 256, 256])
out = unet(in_)
net = eddl.Model([in_],[out])

bin_acc = BinaryAccuracy()
dice = Dice()

net.build(
    eddl.adam(0.00001),
    [eddl.getLoss("binary_cross_entropy")],
    [bin_acc],
    eddl.CS_GPU(mem="low_mem") # Much faster with GPU if available
)

print("INFO - Loading test data")
ts_x = Tensor.load(MICCAI_PATH + "bin/miccai_trX_oriented.bin")
ts_y = Tensor.load(MICCAI_PATH + "bin/miccai_trY_oriented.bin")

for model_idx in range(128):
    print("INFO - Loading model "+str(model_idx))
    eddl.load(net, MODEL_PATH+str(model_idx)+".bin")

    print("INFO - Evaluating model "+str(model_idx))
    os.system(PROFILE_CMD + str(model_idx) + '.csv &')
    eddl.evaluate(net, [ts_x], [ts_y], bs=8)
    os.system('kill $(ps aux | grep nvidia-smi | awk \'{print $2}\')')

print("INFO - Finished")


