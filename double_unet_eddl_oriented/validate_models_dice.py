import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from pyeddl._core  import Metric
from double_unet import double_unet


NUM_MODELS = 35
MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"
THRESHOLD = 0.7

class DiceMetric(Metric):

    def __init__(self):
        Metric.__init__(self, "py_dice")

    def value(self, t, y):        
        t_np = t.getdata() < THRESHOLD
        y_np = y.getdata().astype(np.bool)
        
        union = t_np.sum() + y_np.sum()
        intersection = np.logical_and(t_np, y_np).sum()

        if union == 0:
            return 4
        else:
            return 8*intersection / union

        return dice

print("INFO - Building Dobule U-Net")
in_ = eddl.Input([1, 256, 256])
out = double_unet(in_)
net = eddl.Model([in_],[out])
dice = DiceMetric()

net.build(
    eddl.adam(0.00001),
    [eddl.getLoss("binary_cross_entropy")],
    [dice],
    eddl.CS_GPU(g=[1,0], mem="low_mem")
)

print("INFO - Loading test data")
ts_x = Tensor.load(MICCAI_PATH + "bin/miccai_tsX_oriented.bin")
ts_y = Tensor.load(MICCAI_PATH + "bin/miccai_tsY_oriented.bin")

#print("INFO - Loading train data")
#tr_x = Tensor.load(MICCAI_PATH + "bin/miccai_trX_preprocessed.bin")
#tr_y = Tensor.load(MICCAI_PATH + "bin/miccai_trY_preprocessed.bin")

for i in range(24,NUM_MODELS):
    model_path = "/scrap/users/blazquez/train_oriented/models/double_unet_miccai_"+str(i)+".bin"

    print("INFO - Loading model "+str(i))
    eddl.load(net, model_path)

    print("INFO - Evaluate over test data")
    eddl.evaluate(net, [ts_x], [ts_y], bs=4)

    #print("INFO - Evaluate over train data")
    #eddl.evaluate(net, [tr_x], [tr_y], bs=4)

print("INFO - Finished")
