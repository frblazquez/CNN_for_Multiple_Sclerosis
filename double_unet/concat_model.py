# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# Model to check if several concatenations can be done with PyEDDL


import pyeddl.eddl as eddl

in1 = eddl.Input((256, 1, 1))
in2 = eddl.Input((256, 256, 256))

layer = in1
shape = layer.output.shape[-1]

while shape > 1:
    shape = shape // 2
    layer = eddl.Concat([layer, layer], axis=2)
    layer = eddl.Concat([layer, layer], axis=3)

out = eddl.Mult(in2, layer)
net = eddl.Model([in1, in2], [out])

eddl.build(
    net,
    eddl.nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004), 
    ["binary_cross_entropy"],
    ["dice"],
    eddl.CS_CPU()
)

eddl.summary(net)
