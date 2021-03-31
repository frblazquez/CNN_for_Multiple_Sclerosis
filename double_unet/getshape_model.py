# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# DeepHealth project
#
# Description:
# Model to check problem getting shape with PyEDDL 0.14.0


import pyeddl.eddl as eddl


# I have a model (VGG_19) but I only want up to a certain layer
def model(in_):
    layer = in_

    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 64, [3, 3], [1, 1]), True))
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(layer, 64, [3, 3], [1, 1]), True), "DESIRED_LAYER")
    out = eddl.MaxPool(layer)

    # This print works
    # print(layer.output.shape)

    return eddl.Model([in_], [out])

# In this function I take that layer
def getDesiredLayer(in_):
    net = model(in_)
    layer = eddl.getLayer(net, "DESIRED_LAYER")

    # This print also works
    # print(layer.output.shape)

    return layer

# When I call my function to get that layer it's not initialized
in_ = eddl.Input((3, 256, 256))
layer = getDesiredLayer(in_)

# This print DOESN'T work
print(layer.output.shape)

