import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import warnings
#warnings.simplefilter("ignore")

# class Rotate_batch_Iterator(BatchIterator):
#     """
#     handle class for on-the-fly data augmentation on batches. 
#     Applying 90,180 and 270 degrees rotations and flipping
#     """
#     def transform(self, Xb, yb):
#         Xb, yb = super(Rotate_batch_Iterator, self).transform(Xb, yb)

#         # Flip a given percentage of the images at random:
#         bs = Xb.shape[0]
#         indices = np.random.choice(bs, bs / 2, replace=False)
#         x_da = Xb[indices]
    
#         # apply rotation to the input batch
#         rotate_90 = x_da[:,:,:,::-1,:].transpose(0,1,2,4,3)
#         rotate_180 = rotate_90[:,:,:,::-1,:].transpose(0,1,2,4,3)

#         # apply flipped versions of rotated patches
#         rotate_0_flipped = x_da[:,:,:,:,::-1]
#         rotate_180_flipped = rotate_180[:,:,:,:,::-1]

#         augmented_x = np.stack([rotate_180,
#                                 rotate_0_flipped,
#                                 rotate_180_flipped],
#                                 axis=1)

#         # select random indices from computed transformations
#         #r_indices = np.random.randint(0,7,size=augmented_x.shape[0])
#         r_indices = np.random.randint(0,3,size=augmented_x.shape[0])

#         Xb[indices] = np.stack([augmented_x[i,r_indices[i],:,:,:,:] for i in range(augmented_x.shape[0])])
        
#         return Xb, yb


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
    model1 = models.Sequential()
    model1.add(layers.Conv3D(32, kernel_size=3, activation='relu', padding = 'same', input_shape=(options['patch_size'][0], options['patch_size'][1],options['patch_size'][2], channels)))
    model1.add(layers.BatchNormalization())
    model1.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
    model1.add(layers.Conv3D(64, kernel_size=3, activation='relu', padding = 'same'))
    model1.add(layers.BatchNormalization())
    model1.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
    model1.add(layers.Flatten())
    model1.add(layers.Dropout(rate=0.5))
    model1.add(layers.Dense(256, activation='relu'))
    model1.add(layers.Dense(2)) #add softmax activation or not?
    # TODO: Try adding here sigmoid    
    
    model1.compile(optimizer='adadelta',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #should this be binary? from logits or not?
              metrics=['accuracy'])
   
    # --------------------------------------------------
    # second model
    # --------------------------------------------------
    
    model2 = models.Sequential()
    model2.add(layers.Conv3D(32, kernel_size=3, activation='relu', padding = 'same', input_shape=(options['patch_size'][0], options['patch_size'][1],options['patch_size'][2],channels)))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
    model2.add(layers.Conv3D(64, kernel_size=3, activation='relu', padding = 'same'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
    model2.add(layers.Flatten())
    model2.add(layers.Dropout(rate=0.5))
    model2.add(layers.Dense(256, activation='relu'))
    model2.add(layers.Dense(2))
    # TODO: Try adding here sigmoid
    
    model2.compile(optimizer='adadelta',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #should this be binary? from logits or not?
              metrics=['accuracy'])
    

#     # upload weights if set
#     if options['load_weights'] == 'True':
#         print "    --> CNN, loading weights from", options['experiment'], 'configuration'
#         net1.load_params_from(net_weights)
#         net2.load_params_from(net_weights2)
    return [model1, model2]

