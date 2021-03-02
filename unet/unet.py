# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# U-Net implementation using PyEDDL library
#
# References:
# https://arxiv.org/abs/1505.04597
# https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical


import pyeddl.eddl as eddl


# U-Net implementation
#
# References:
# https://arxiv.org/abs/1505.04597
# https://github.com/zhixuhao/unet/blob/master/model.py
# https://github.com/deephealthproject/pyeddl/blob/master/examples/_OLD/eddl_unet.py
def unet(in_):
    
    # Encoding
    conv1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_  , 64, [3, 3], [1, 1]), True))
    conv1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv1, 64, [3, 3], [1, 1]), True))
    conv2 = eddl.MaxPool(conv1)

    conv2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv2, 128, [3, 3], [1, 1]), True))
    conv2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv2, 128, [3, 3], [1, 1]), True))
    conv3 = eddl.MaxPool(conv2)

    conv3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv3, 256, [3, 3], [1, 1]), True))
    conv3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv3, 256, [3, 3], [1, 1]), True))
    conv4 = eddl.MaxPool(conv3)

    conv4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv4, 512, [3, 3], [1, 1]), True))
    conv4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv4, 512, [3, 3], [1, 1]), True))
    conv4 = eddl.Dropout(conv4, 0.5)
    conv5 = eddl.MaxPool(conv4)

    conv5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv5, 1024, [3, 3], [1, 1]), True))
    conv5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv5, 1024, [3, 3], [1, 1]), True))
    conv5 = eddl.Dropout(conv5, 0.5)
    
    # Decoding
    conv6 = eddl.UpSampling(conv5, [2,2])
    conv6 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv6, 512, [2, 2], [1, 1]), True))
    conv6 = eddl.Concat([conv4, conv6])
    conv6 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv6, 512, [3, 3], [1, 1]), True))
    conv6 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv6, 512, [3, 3], [1, 1]), True))

    conv7 = eddl.UpSampling(conv6, [2,2])
    conv7 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv7, 256, [2, 2], [1, 1]), True))
    conv7 = eddl.Concat([conv3, conv7])
    conv7 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv7, 256, [3, 3], [1, 1]), True))
    conv7 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv7, 256, [3, 3], [1, 1]), True))

    conv8 = eddl.UpSampling(conv7, [2,2])
    conv8 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv8, 128, [2, 2], [1, 1]), True))
    conv8 = eddl.Concat([conv2, conv8])
    conv8 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv8, 128, [3, 3], [1, 1]), True))
    conv8 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv8, 128, [3, 3], [1, 1]), True))

    conv9 = eddl.UpSampling(conv8, [2,2])
    conv9 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv9,  64, [2, 2], [1, 1]), True))
    conv9 = eddl.Concat([conv1, conv9])
    conv9 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv9,  64, [3, 3], [1, 1]), True))
    conv9 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(conv9,  64, [3, 3], [1, 1]), True))

    # Output
    out = eddl.ReLu(   eddl.BatchNormalization(eddl.Conv(conv9, 2, [3, 3], [1, 1]), True))
    out = eddl.Sigmoid(eddl.BatchNormalization(eddl.Conv(out  , 1, [1, 1], [1, 1]), True))

    return eddl.Model([in_], [out])





# Double U-Net construction test
if __name__ == "__main__":
    model = unet(eddl.Input((3,256, 256)))

    eddl.build(
        model,
        eddl.nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004), # Consider modifying 'schedule_decay'
        ["binary_cross_entropy"],
        ["dice"],
        eddl.CS_CPU()
    )

    eddl.summary(model)
