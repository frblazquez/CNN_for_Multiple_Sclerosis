# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Double U-Net implementation using PyEDDL library
#
# References:
# https://arxiv.org/pdf/2006.04868.pdf
# https://github.com/DebeshJha/2020-CBMS-DoubleU-Net


import pyeddl.eddl as eddl


# Names for different VGG19 layers 
VGG19_BLOCK_1 = "vgg19_block_1"
VGG19_BLOCK_2 = "vgg19_block_2"
VGG19_BLOCK_3 = "vgg19_block_3"
VGG19_BLOCK_4 = "vgg19_block_4"
VGG19_BLOCK_5 = "vgg19_block_5"


# VGG19 implementation
#
# References:
# https://iq.opengenus.org/vgg19-architecture/
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py
# https://github.com/deephealthproject/pyeddl/blob/master/examples/NN/2_CIFAR10/cifar_vgg16.py
def vgg19(in_, num_classes=2, include_top=False):
    layer = in_

    # There is no batch normalization in the Keras implementation, but there is in EDDL's VGG16!
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 64, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 64, [3, 3], [1, 1]), True), VGG19_BLOCK_1)
    layer = eddl.MaxPool(layer)

    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 128, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 128, [3, 3], [1, 1]), True), VGG19_BLOCK_2)
    layer = eddl.MaxPool(layer)

    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 256, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 256, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 256, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 256, [3, 3], [1, 1]), True), VGG19_BLOCK_3)
    layer = eddl.MaxPool(layer)

    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True), VGG19_BLOCK_4)
    layer = eddl.MaxPool(layer)

    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 512, [3, 3], [1, 1]), True), VGG19_BLOCK_5)
    out = eddl.MaxPool(layer)

    if include_top:
        out = eddl.Flatten(out)

        out = eddl.Activation(eddl.Dense(out, 4096), "relu")
        out = eddl.Activation(eddl.Dense(out, 4096), "relu")    
        out = eddl.Softmax(eddl.Dense(out, num_classes))

    return eddl.Model([in_], [out])


# ASPP implementation
#
# References:
# https://www.paperswithcode.com/method/aspp
# https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/model.py
# https://github.com/deephealthproject/eddl/blob/master/examples/nn/3_drive/1_drive_seg.cpp
def aspp(in_, filter):
    shape = in_.output.shape

    l1 = eddl.AveragePool(in_, [shape[2], shape[3]], [shape[2], shape[3]], padding="valid")    
    l1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(l1 , filter, [1, 1], use_bias=True , dilation_rate=[1,1]),   True))
    l2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [1, 1], use_bias=False, dilation_rate=[1,1]),   True))
    l3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [3, 3], use_bias=False, dilation_rate=[6,6]),   True))
    l4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [3, 3], use_bias=False, dilation_rate=[12,12]), True))
    l5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [3, 3], use_bias=False, dilation_rate=[18,18]), True))
    l1 = eddl.UpSampling(l1, [shape[2], shape[3]], interpolation="bilinear")

    out= eddl.Concat([l1, l2, l3, l4, l5])
    out= eddl.ReLu(eddl.BatchNormalization(eddl.Conv(in_, filter, [1, 1], use_bias=False, dilation_rate=[1,1]),   True))

    return out


def squeeze_excite_block(in_, ratio=8):
    init = in_
    filters = init.output.shape[1] # Take the number of channels
    
    out = eddl.GlobalAveragePool(init)
    out = eddl.Reshape(out, [filters])
    
    # kernel_initializer='he_normal' option not available!
    out = eddl.ReLu(eddl.Dense(out, filters//ratio, use_bias=False))
    out = eddl.Sigmoid(eddl.Dense(out, filters, use_bias=False))
    
    # Reshaping/Concatenating for doing Mult(init, out)
    shape = init.output.shape[-1]
    out = eddl.Reshape(out, [filters,1,1])

    while shape != 1:
        shape = shape // 2
        out = eddl.Concat([out, out], axis=2)
        out = eddl.Concat([out, out], axis=3)
        
    return eddl.Mult(init, out)


def decoder1(in_, vgg19_skips):
    decoder_out = in_
    filters = [256, 128, 64, 32]
    
    for i, filter in enumerate(filters):
        decoder_out = eddl.UpSampling(decoder_out, [2,2], interpolation="bilinear")
        decoder_out = eddl.Concat([decoder_out, vgg19_skips[i]])
        
        # conv_block in 2020-CBMS-DoubleU-Net
        decoder_out = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(decoder_out, filter, [3, 3], [1, 1]), True))
        decoder_out = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(decoder_out, filter, [3, 3], [1, 1]), True))
        decoder_out = squeeze_excite_block(decoder_out) 
    
    return eddl.Sigmoid(eddl.Conv(decoder_out, 1, [1, 1], [1, 1]))


def decoder2(in_, vgg19_skips, encoder_skips):
    decoder_out = in_
    filters = [256, 128, 64, 32]
    
    for i, filter in enumerate(filters):
        decoder_out = eddl.UpSampling(decoder_out, [2,2], interpolation="bilinear")
        decoder_out = eddl.Concat([decoder_out, vgg19_skips[i], encoder_skips[i]])
        
        # conv_block in 2020-CBMS-DoubleU-Net
        decoder_out = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(decoder_out, filter, [3, 3], [1, 1]), True))
        decoder_out = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(decoder_out, filter, [3, 3], [1, 1]), True))
        decoder_out = squeeze_excite_block(decoder_out)
    
    return eddl.Sigmoid(eddl.Conv(decoder_out, 1, [1, 1], [1, 1]))
    
def encoder1(in_):
    vgg19_net = vgg19(in_, include_top=False)
    vgg19_out = eddl.getLayer(vgg19_net, VGG19_BLOCK_5)
    
    vgg19_block1_out = eddl.getLayer(vgg19_net, VGG19_BLOCK_1)
    vgg19_block2_out = eddl.getLayer(vgg19_net, VGG19_BLOCK_2)
    vgg19_block3_out = eddl.getLayer(vgg19_net, VGG19_BLOCK_3)
    vgg19_block4_out = eddl.getLayer(vgg19_net, VGG19_BLOCK_4)
    vgg19_blocks_out = [vgg19_block4_out, vgg19_block3_out, vgg19_block2_out, vgg19_block1_out]

    return vgg19_out, vgg19_blocks_out

def encoder2(in_):
    encoder_out   = in_
    filters       = [32, 64, 128, 256]
    block_outputs = []

    for i, filter in enumerate(filters):
        # conv_block in 2020-CBMS-DoubleU-Net
        encoder_out = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(encoder_out, filter, [3, 3], [1, 1]), True))
        encoder_out = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(encoder_out, filter, [3, 3], [1, 1]), True))
        encoder_out = squeeze_excite_block(encoder_out)
        
        block_outputs.insert(0, encoder_out)
        encoder_out = eddl.MaxPool(encoder_out)

    return encoder_out, block_outputs



# Double U-Net implementation
#
# References:
# https://paperswithcode.com/paper/doubleu-net-a-deep-convolutional-neural
# https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/img/DoubleU-Net.png
# https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/model.py
def double_unet(in_):

    # Encode 1-ASPP-Decode 1, first U-Net
    vgg19_out, vgg19_blocks_out = encoder1(in_)
    aspp1_out = aspp(vgg19_out, 64)
    output1   = decoder1(aspp1_out, vgg19_blocks_out)
    
    # Encode 2-ASPP-Decode 2, second U-Net
    multiply  = eddl.Mult(in_, output1)
    encoder_out, encoder_blocks_out = encoder2(multiply)
    aspp2_out = aspp(encoder_out, 64)
    output2   = decoder2(aspp2_out, vgg19_blocks_out, encoder_blocks_out)
    
    # Double U-Net output if we want to have output1 also
    # out = eddl.Concat([output1, output2])
    
    return eddl.Model([in_], [output2])






# Double U-Net construction test
if __name__ == "__main__":
    model = double_unet(eddl.Input((3,256, 256)))

    eddl.build(
        model,
        eddl.nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004), 
        ["binary_cross_entropy"],
        ["dice"],
        eddl.CS_CPU()
    )

    eddl.summary(model)
