#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import warnings
from custom_conv3d import conv3d #TODO: Import 3D layers from pyeddl

def cascade_model(options):
    """
    3D cascade model using Tensorflow
    
    Inputs:
    - model_options:
    - weights_path: path to where weights should be saved

    Output:
    - nets = list of NeuralNets (CNN1, CNN2)
    """

    # model options
    channels = len(options['modalities'])
    train_split_perc = options['train_split']
    num_epochs = options['max_epochs']
    max_epochs_patience = options['patience']

    
    # save model to disk to re-use it. Create an experiment folder
    # organize experiment 
    if not os.path.exists(os.path.join(options['weight_paths'], options['experiment'])):
        os.mkdir(os.path.join(options['weight_paths'], options['experiment']))
    if not os.path.exists(os.path.join(options['weight_paths'], options['experiment'], 'nets')):
        os.mkdir(os.path.join(options['weight_paths'], options['experiment'], 'nets'))


    # --------------------------------------------------
    # first model
    # --------------------------------------------------
    in_ = eddl.Input([3, 11, 11, 11])
    layer = in_
    #TODO: replace conv3d layer with eddl conv3d followed by a batch normalization layer followed by maxpool3d layer.
    layer = eddl.ReLu(conv3d(layer, 32, [3,3,3], name='conv3d_1'))
    layer = eddl.ReLu(conv3d(layer, 64, [3,3,3], name='conv3d_2'))
    
    layer = eddl.Flatten(layer)
    layer = eddl.Dropout(layer, 0.5)
    layer = eddl.ReLu(eddl.GlorotUniform(eddl.Dense(layer, 256)))
    out = eddl.GlorotUniform(eddl.Dense(layer, 1))
    model1 = eddl.Model([in_], [out])

    eddl.build(
    model1,
    #TODO: check if adadelta is now available in eddl new version; if yes replace adam.
    #      adadelta is there but don't trust it!
    eddl.adam(0.0001),#adadelta(0.001, 0.95, 1e-4, 0),
    ["binary_cross_entropy"],
    ["binary_accuracy"],
    eddl.CS_CPU(th=4)
    )
    print('first model generated successfully!')
    
    # --------------------------------------------------
    # second model
    # --------------------------------------------------
    eddl.plot(model1, "model1new.pdf")
    in_ = eddl.Input([3, 11, 11, 11])
    layer = in_
    #TODO: replace conv3d layer with eddl conv3d followed by a batch normalization layer followed by maxpool3d layer.
    layer = eddl.ReLu(conv3d(layer, 32, [3,3,3], name='Conv3d_3'))
    layer = eddl.ReLu(conv3d(layer, 64, [3,3,3], name='Conv3d_4'))
    
    layer = eddl.Flatten(layer)
    layer = eddl.Dropout(layer, 0.5)
    layer = eddl.ReLu(eddl.GlorotUniform(eddl.Dense(layer, 256)))
    out = eddl.GlorotUniform(eddl.Dense(layer, 1))
    model2 = eddl.Model([in_], [out])

    eddl.build(
    model2,
    #TODO: check if adadelta is now available in eddl new version; if yes replace adam.
    #      adadelta is there but don't trust it!
    eddl.adam(0.0001),#adadelta(0.001, 0.95, 1e-4, 0),
    ["binary_cross_entropy"],
    ["binary_accuracy"],
    eddl.CS_CPU(th=16)
    )
    print('2nd model generated successfully!')
    return [model1, model2]

