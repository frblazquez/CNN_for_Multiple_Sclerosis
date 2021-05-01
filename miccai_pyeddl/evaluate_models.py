import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from pyeddl._core  import Metric
from double_unet import double_unet


NUM_MODELS = 80
MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"
THRESHOLD = 0.5

class DiceMetric(Metric):

    def __init__(self):
        Metric.__init__(self, "py_dice_metric")

    def value(self, t, y):
        if t.shape != y.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        t_np = t.getdata() > THRESHOLD
        y_np = y.getdata().astype(np.bool)

        union = t_np.sum() + y_np.sum()
        intersection = np.logical_and(~t_np, ~y_np).sum()
        #intersection = np.logical_and(t_np, y_np).sum()

        # Thanks to doing logical negation for the intersection this shouldn't be necessary!
        if union == 0:
            return 0

        return 2 * intersection / union



print("INFO - Building Dobule U-Net")
in_ = eddl.Input([1, 256, 256])
out = double_unet(in_)
net = eddl.Model([in_],[out])

dice = DiceMetric()

eddl.build(
    net,
    eddl.adam(0.00001),
    ["binary_cross_entropy"],
    [dice],
    eddl.CS_GPU(mem="low_mem")
)

print("INFO - Loading test data")
ts_x = Tensor.load(MICCAI_PATH + "bin/miccai_tsX_preprocessed.bin")
ts_y = Tensor.load(MICCAI_PATH + "bin/miccai_tsY_preprocessed.bin")

for i in range(NUM_MODELS):
    model_path = "/scrap/users/blazquez/train/models/double_unet_miccai_"+str(i)+".bin"

    print("INFO - Loading model "+str(i))
    eddl.load(net, model_path)

    print("INFO - Evaluate over test data")
    eddl.evaluate(net, [ts_x], [ts_y], bs=16)

print("INFO - Finished")
