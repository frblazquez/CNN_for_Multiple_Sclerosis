#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyeddl import eddl
import os
from base_eddl_new import *
from build_model_eddl_new import cascade_model
#from config import *
import base_eddl_new
import numpy as np
import tracemalloc
import time

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
options['experiment'] = 'test_CNN'

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

options['max_epochs'] = 1#100# 500

# maximum number of epochs without improving validation before stopping training (early stopping) 
#options['patience'] = 25
options['patience'] = 10

# Number of samples used to test at once. This parameter should be around 50000 for machines
# with less than 32GB of RAM
options['batch_size'] = 512
#options['batch_size'] = 32

# net print verbosity. Set to zero for this particular notebook, but setting this value to 11 is recommended
options['net_verbose'] = 1

# post-processing binary threshold. After segmentation, probabilistic masks are binarized using a defined threshold.
options['t_bin'] = 0.8

# The resulting binary mask is filtered by removing lesion regions with lesion size before a defined value
options['l_min'] = 5


# ## Experiment configuration:
# 
# Organize the experiment. Although not necessary, intermediate results, network weights and final lesion segmentation masks are stored inside a folder with name `options['experiment']`. This is extremely useful when a lot of experiments are computed on the same images to declutter the user space. 

exp_folder = os.path.join(os.getcwd(), options['experiment'])
if not os.path.exists(exp_folder):
    os.mkdir(exp_folder)
    os.mkdir(os.path.join(exp_folder,'nets'))
    os.mkdir(os.path.join(exp_folder,'.train'))

# set the output name 
options['test_name'] = 'cnn_' + options['experiment'] + '.nii.gz'
 
    
#Open a file for performance logging
fh = open("train_logging_eddl/train_stats_full_system_memory.txt","w+")

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


train_folder = 'miccai2016/'
train_mask_folder = 'miccai2016_unprocessed/'
train_x_data = {}
train_y_data = {}

# TRAIN X DATA
DEBUG = False
if DEBUG:
    train_vec = [1]
else:
    train_vec = [0,1,2,3,4,5,6,7,8,10,11,12]
test_vec = [9,13,14]
for i in train_vec:
    subj_name = 's' + str(i+1)
    if i<5:
        train_x_data[subj_name] = {'FLAIR': train_folder + subj_name + '/FLAIR_preprocessed_downsampled.nii.gz' ,
    #                         'DP': train_folder + subj_name + '/DP_preprocessed.nii.gz',
    #                         'GADO': train_folder + subj_name +  '/GADO_preprocessed.nii.gz', 
                             'T1': train_folder + subj_name +  '/T1_preprocessed_downsampled.nii.gz',
                            'T2': train_folder + subj_name +  '/T2_preprocessed_downsampled.nii.gz'}
    #                              }
        train_y_data[subj_name] = train_mask_folder + subj_name +  '/Consensus_downsampled.nii.gz'
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

# In[ ]:



options['weight_paths'] = os.getcwd()
model = cascade_model(options)
eddl.summary(model[0])
eddl.summary(model[1])


# ## Train the model:
# 
# The function `train_cascaded_model` is used to train the model. The next image summarizes the training procedure. For further information about how this function optimizes the two CNN, please consult the original paper. (**NOTE**: For this example, `options['net_verbose`] has been set to `0` for simplicity)
# 
# 
# ![](pipeline_training.png)
# 
# 

# In[ ]:

tracemalloc.start()


model = train_cascaded_model(model, train_x_data, train_y_data,  options)

current, peak = tracemalloc.get_traced_memory()
fh.write("TRAINING: Current memory usage is {0} MB; Peak was {1} MB\n".format(current / 10**6,peak / 10**6))
tracemalloc.stop()



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


# TEST X DATA
test_x_data = {}
test_y_data = {}
for i in test_vec:
    start = time.time()
    subj_name = 's' + str(i+1)
    print("Testing subject" + subj_name)
    test_x_data[subj_name] = {'FLAIR': train_folder + subj_name + '/FLAIR_preprocessed.nii.gz', 
    #                         'DP': train_folder + subj_name + '/DP_preprocessed.nii.gz',
    #                         'GADO': train_folder + subj_name +  '/GADO_preprocessed.nii.gz', 
                              'T1': train_folder + subj_name +  '/T1_preprocessed.nii.gz',
                            'T2': train_folder + subj_name +  '/T2_preprocessed.nii.gz'
                          }
    #                              


    options['test_folder'] = train_folder
    options['test_scan'] = subj_name
    out_seg = test_cascaded_model(model, test_x_data, options)

    out_file_name = 'results_eddl/' + subj_name + '_output_segmentation_eddl.npy'
    np.save(out_file_name,out_seg)
    test_x_data = {}
    print(f'inference time:{time.time()-start:.2f}')

fh.close()
