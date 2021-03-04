# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# VGG19 implementation using EDDL library
#
# References:
# https://iq.opengenus.org/vgg19-architecture/
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py
# https://github.com/deephealthproject/pyeddl/blob/master/examples/NN/2_CIFAR10/cifar_vgg16.py

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

# Names for different VGG19 layers 
VGG19_BLOCK_1 = "vgg19_block_1"
VGG19_BLOCK_2 = "vgg19_block_2"
VGG19_BLOCK_3 = "vgg19_block_3"
VGG19_BLOCK_4 = "vgg19_block_4"
VGG19_BLOCK_5 = "vgg19_block_5"


def vgg19(in_, num_classes=2):
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
    layer = eddl.MaxPool(layer)

    layer = eddl.Flatten(layer)

    layer = eddl.Activation(eddl.Dense(layer, 4096), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 4096), "relu")    
    out   = eddl.Softmax(eddl.Dense(layer, num_classes))

    return eddl.Model([in_], [out])

def main():

    PATH_TO_CIFAR10 = "/home/francisco/Documents/Universidad/5º_Carrera/TFG_Computer_Science/datasets/cifar10/"
    epochs = 5
    batch_size = 8
    num_classes = 10    

    in_ = eddl.Input([3,32,32])
    net = vgg19(in_, num_classes)
    
    eddl.build(
        net,
        eddl.rmsprop(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU()
    )
    
    x_train = Tensor.load(PATH_TO_CIFAR10 + "cifar_trX.bin")
    y_train = Tensor.load(PATH_TO_CIFAR10 + "cifar_trY.bin")
    x_test  = Tensor.load(PATH_TO_CIFAR10 + "cifar_tsX.bin")
    y_test  = Tensor.load(PATH_TO_CIFAR10 + "cifar_tsY.bin")
    x_train.div_(255.0)
    x_test.div_(255.0)

    eddl.fit(net, [x_train], [y_train], batch_size, epochs)
    eddl.evaluate(net, [x_test], [y_test])

if __name__ == "__main__":
    main()

