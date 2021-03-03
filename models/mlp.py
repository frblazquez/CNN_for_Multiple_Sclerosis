# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Simple MLP implementation using EDDL library
#
# References:
# https://deephealthproject.github.io/pyeddl/getting_started.html

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

def mlp(in_, num_layers, width, num_classes):
    layer = in_
    for _ in range(num_layers):
    	layer = eddl.LeakyReLu(eddl.Dense(layer, width))
    out = eddl.Softmax(eddl.Dense(layer, num_classes))
    
    return eddl.Model([in_], [out])
    

def main():

    PATH_TO_MNIST = "/home/francisco/Documents/Universidad/5º_Carrera/TFG_Computer_Science/datasets/mnist/"
    epochs = 5
    batch_size = 100
    num_classes = 10

    in_ = eddl.Input([784])
    net = mlp(in_, 3, 1024, 10)
    
    eddl.build(
        net,
        eddl.rmsprop(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU()
    )
    
    x_train = Tensor.load(PATH_TO_MNIST + "mnist_trX.bin")
    y_train = Tensor.load(PATH_TO_MNIST + "mnist_trY.bin")
    x_test = Tensor.load(PATH_TO_MNIST + "mnist_tsX.bin")
    y_test = Tensor.load(PATH_TO_MNIST + "mnist_tsY.bin")
    x_train.div_(255.0)
    x_test.div_(255.0)

    eddl.setlogfile(net, "1_mlp.log")
    eddl.summary(net)

    eddl.fit(net, [x_train], [y_train], batch_size, epochs)
    eddl.evaluate(net, [x_test], [y_test])
    
    eddl.save(net, "1_mlp.bin")
    eddl.save_net_to_onnx_file(net, "1_mlp.onnx")

if __name__ == "__main__":
    main()

