#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import warnings

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
    
    # TODO: Batch normalization not working in 3d, not added before max pooling!
    #layer = eddl.MaxPool3D(eddl.BatchNormalization(layer, True), padding='valid')
    
    in_ = eddl.Input([3, 11, 11, 11])
    layer = in_
    layer = eddl.ReLu(eddl.Conv3D(layer, 32, [3,3,3], name='conv3d_1')) # Input_shape? Not required
    layer = eddl.MaxPool3D(layer, padding='valid')
    layer = eddl.ReLu(eddl.Conv3D(layer, 64, [3,3,3], name='conv3d_2')) 
    layer = eddl.MaxPool3D(layer, padding='valid')

    layer = eddl.Flatten(layer)
    layer = eddl.Dropout(layer, 0.5)
    layer = eddl.ReLu(eddl.GlorotUniform(eddl.Dense(layer, 256)))
    out = eddl.GlorotUniform(eddl.Dense(layer, 1))
    # TODO: Add sigmoid here to be able to compute probabilities!

    model1 = eddl.Model([in_], [out])

    eddl.build(
    model1,
    eddl.adadelta(0.001, 0.95, 1e-4, 0),
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
    layer = eddl.ReLu(eddl.Conv3D(layer, 32, [3,3,3], name='Conv3d_3')) # Input_shape? Not required
    layer = eddl.MaxPool3D(layer, padding='valid')
    layer = eddl.ReLu(eddl.Conv3D(layer, 64, [3,3,3], name='Conv3d_4')) 
    layer = eddl.MaxPool3D(layer, padding='valid')
    
    layer = eddl.Flatten(layer)
    layer = eddl.Dropout(layer, 0.5)
    layer = eddl.ReLu(eddl.GlorotUniform(eddl.Dense(layer, 256)))
    out = eddl.GlorotUniform(eddl.Dense(layer, 1))
    # TODO: Add sigmoid here to be able to compute probabilities!

    model2 = eddl.Model([in_], [out])

    eddl.build(
    model2,
    eddl.adadelta(0.001, 0.95, 1e-4, 0),
    ["binary_cross_entropy"],
    ["binary_accuracy"],
    eddl.CS_CPU(th=16)
    )
    print('2nd model generated successfully!')
    return [model1, model2]

