#!/usr/bin/env python
# coding: utf-8

# # Multiple Sclerosis (MS) lesion segmentation of MRI images using a cascade of two 3D convolutional neural networks 
# Training network on 1 downsampled image (3 imaging modalities) from the MSSEG 2016 database for 5 epochs for performance profiling
# Use 5 cores when testing in order to get consistent time/memory results

# Import libraries: 

import os
from collections import OrderedDict
from base import *
from build_model_tf import cascade_model
from config import *
import base
import numpy as np
import tracemalloc

# ## Model configuration:
# Configure the model options. Options are passed to the model using the dictionary `options`. The main options are:

options = {}

# --------------------------------------------------
# Experiment parameters
# --------------------------------------------------

# image modalities used (T1, FLAIR, PD, T2, ...) 
#options['modalities'] = ['T1', 'FLAIR','GADO','DP','T2']
options['modalities'] = ['FLAIR','T1','T2']

# Select an experiment name to store net weights and segmentation masks
options['experiment'] = 'train_short_system'

# In order to expose the classifier to more challeging samples, a threshold can be used to to select 
# candidate voxels for training. Note that images are internally normalized to 0 mean 1 standard deviation 
# before applying thresholding. So a value of t > 0.5 on FLAIR is reasonable in most cases to extract 
# all WM lesion candidates
options['min_th'] = 0.5

# randomize training features before fitting the model.  
options['randomize_train'] = True

# Select between pixel-wise or fully-convolutional training models. Although implemented, fully-convolutional
# models have been not tested with this cascaded model 
options['fully_convolutional'] = False


# --------------------------------------------------
# model parameters
# --------------------------------------------------

# 3D patch size. So, far only implemented for 3D CNN models. 
options['patch_size'] = (11,11,11)

# percentage of the training vector that is going to be used to validate the model during training
options['train_split'] = 0.25

# maximum number of epochs used to train the model
#options['max_epochs'] = 200

options['max_epochs'] = 5

# maximum number of epochs without improving validation before stopping training (early stopping) 
#options['patience'] = 25
options['patience'] = 1

# Number of samples used to test at once
options['batch_size'] = 512

# net print verbosity
options['net_verbose'] = 1

# post-processing binary threshold. After segmentation, probabilistic masks are binarized using a defined threshold.
options['t_bin'] = 0.8

# The resulting binary mask is filtered by removing lesion regions with lesion size before a defined value
options['l_min'] = 5


# ## Experiment configuration:
# 
# Organize the experiment. Although not necessary, intermediate results, network weights and final lesion segmentation masks are stored inside a folder with name `options['experiment']`. This is extremely useful when a lot of experiments are computed on the same images to declutter the user space. 
 
    
#Open a file for performance logging
fh = open("train_logging/" + options['experiment'] + "_memory.txt","w+")

# ## Load the training data:
# 
# Training data is internally loaded by the method. So far, training and testing images are passed as dictionaries, where each training image is stored as follows: 
# 
# ```
# traininig_X_data['image_identifier'] = {'modality_1': /path/to/image_modality_n.nii.gz/,
#                                          ....
#                                         'modality_n': /path/to/image_modality_n.nii.gz/}
# ```
# 
# And also for labels: 
# 
# ```
# traininig_y_data['image_identifier_1'] = 'path/to/image_labels.nii.gz/'
# ```
# 
# **NOTE**: As stated in the paper, input images have been already skull-stripped and bias corrected (N3, etc...) by the user before running the classifer.

# In[14]:


train_folder = 'data/miccai2016/'
train_mask_folder = 'data/miccai2016_unprocessed/'
train_x_data = {}
train_y_data = {}

# TRAIN X DATA
train_vec = [0]
for i in train_vec:
    subj_name = 's' + str(i+1)
    #Downsampled images
    if i<5:
        train_x_data[subj_name] = {'FLAIR': train_folder + subj_name + '/FLAIR_preprocessed_downsampled.nii.gz' ,
    #                         'DP': train_folder + subj_name + '/DP_preprocessed.nii.gz',
    #                         'GADO': train_folder + subj_name +  '/GADO_preprocessed.nii.gz', 
                             'T1': train_folder + subj_name +  '/T1_preprocessed_downsampled.nii.gz',
                            'T2': train_folder + subj_name +  '/T2_preprocessed_downsampled.nii.gz'}
    #                              }
        train_y_data[subj_name] = train_mask_folder + subj_name +  '/Consensus_downsampled.nii.gz'
    #Normally sized images
    else:
        train_x_data[subj_name] = {'FLAIR': train_folder + subj_name + '/FLAIR_preprocessed.nii.gz' ,
    #                         'DP': train_folder + subj_name + '/DP_preprocessed.nii.gz',
    #                         'GADO': train_folder + subj_name +  '/GADO_preprocessed.nii.gz', 
                             'T1': train_folder + subj_name +  '/T1_preprocessed.nii.gz',
                            'T2': train_folder + subj_name +  '/T2_preprocessed.nii.gz'}
    #                              }
        train_y_data[subj_name] = train_mask_folder + subj_name +  '/Consensus.nii.gz'



# ## Initialize the model:
# 
# The model is initialized using the function `cascade_model`, which returns a list of two `NeuralNet` objects. Optimized weights are stored also inside the experiment folder for future use (testing different images without re-training the model.  


options['weight_paths'] = os.getcwd()
model = cascade_model(options)
model[0].summary()
model[1].summary()


# ## Train the model:
# 
# The function `train_cascaded_model` is used to train the model. The next image summarizes the training procedure. For further information about how this function optimizes the two CNN, please consult the original paper. (**NOTE**: For this example, `options['net_verbose`] has been set to `0` for simplicity)


tracemalloc.start() #memory profiling


model, history1, history2 = train_cascaded_model(model, train_x_data, train_y_data,  options)

current, peak = tracemalloc.get_traced_memory()
fh.write("TRAINING: Current memory usage is {0} MB; Peak was {1} MB\n".format(current / 10**6,peak / 10**6))
print("TRAINING: Current memory usage is {0} MB; Peak was {1} MB\n".format(current / 10**6,peak / 10**6))
tracemalloc.stop()

# In[ ]:




# ## Test the model:
# 
# Once the model has been trained, it can e tested on other images. Please note that the same image modalities have to be used. Testing images are loaded equally to training_data, so a `dictionary` defines the modalities used:
# 
# ```
# test_X_data['image_identifier'] = {'modality_1': /path/to/image_modality_n.nii.gz/,
#                                          ....
#                                    'modality_n': /path/to/image_modality_n.nii.gz/}
# ```
# 

print("Done!")

fh.close()
