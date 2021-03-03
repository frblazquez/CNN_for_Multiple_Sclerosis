# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# ASPP implementation using EDDL library
#
# References:
# https://www.paperswithcode.com/method/aspp
# https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/model.py
# https://github.com/deephealthproject/eddl/blob/master/examples/nn/3_drive/1_drive_seg.cpp


import pyeddl.eddl as eddl


def aspp(in_, filter):

    shape = in_.output.shape
    
    l0 = eddl.AveragePool(in_, [shape[2], shape[3]])
    l1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [1, 1], use_bias=False, dilation_rate=[1,1]),   True))
    l2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [1, 1], use_bias=False, dilation_rate=[1,1]),   True))
    l3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [3, 3], use_bias=False, dilation_rate=[6,6]),   True))
    l4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [3, 3], use_bias=False, dilation_rate=[12,12]), True))
    l5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [3, 3], use_bias=False, dilation_rate=[18,18]), True))
    l1 = eddl.UpSampling(l1, [shape[2], shape[3]], interpolation="bilinear")

    l  = eddl.Concat([l1, l2, l3, l4, l5])
    l  = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [1, 1], use_bias=False, dilation_rate=[1,1]),   True))

    return l

