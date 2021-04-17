# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Drive dataset preprocessing for U-Net model
#
# References:
# https://drive.grand-challenge.org/
# https://github.com/deephealthproject/pyeddl/blob/master/examples/NN/3_DRIVE/drive_seg.py

import os
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt
import skimage.transform as skTrans

# Set the following variable to the folder containing the DRIVE dataset
MICCAI_PATH = "/scrap/users/blazquez/datasets/miccai2016/"


def main():
    # Test images are selected so that there is one of each original shape
    test_imgs_idx  = [4,9,14]

    train_imgs_path = [MICCAI_PATH + 'preprocessed/s'+str(i)+'/FLAIR_preprocessed.nii' for i in range(1,16) if i not in test_imgs_idx]
    train_masks_path= [MICCAI_PATH + 'unprocessed/s'+str(i)+'/Consensus.nii'           for i in range(1,16) if i not in test_imgs_idx]
    test_imgs_path  = [MICCAI_PATH + 'preprocessed/s'+str(i)+'/FLAIR_preprocessed.nii' for i in test_imgs_idx]
    test_masks_path = [MICCAI_PATH + 'unprocessed/s'+str(i)+'/Consensus.nii'           for i in test_imgs_idx]

    # Train data binary files creation
    train_x = []
    train_y = []

    for (img,msk) in zip(train_imgs_path, train_masks_path):
        img_np = nib.load(img).get_fdata()
        msk_np = nib.load(msk).get_fdata()
    
        # First we resize the image
        img_np = skTrans.resize(img_np, (256,256,256), order=1, preserve_range=True)
        msk_np = skTrans.resize(msk_np, (256,256,256), order=1, preserve_range=True)
    
        # Then we slice the 3D image to several 2D images
        for i in range(256):
            train_x.append(img_np[:,:,i])
            train_y.append(msk_np[:,:,i])
            train_x.append(img_np[:,i,:])
            train_y.append(msk_np[:,i,:])
            train_x.append(img_np[i,:,:])
            train_y.append(msk_np[i,:,:])
        
    tensor_x = Tensor.fromarray(train_x)
    tensor_y = Tensor.fromarray(train_y)

    tensor_x.save("miccai_trX_preprocessed.bin", "bin")
    tensor_y.save("miccai_trY_preprocessed.bin", "bin")

    # Train data binary files creation
    test_x = []
    test_y = []
    
    for (img,msk) in zip(test_imgs_path, test_masks_path):
        img_np = nib.load(img).get_fdata()
        msk_np = nib.load(msk).get_fdata()
        
        # First we resize the image
        img_np = skTrans.resize(img_np, (256,256,256), order=1, preserve_range=True)
        msk_np = skTrans.resize(msk_np, (256,256,256), order=1, preserve_range=True)
    
        # Then we slice the 3D image to several 2D images
        for i in range(256):
            test_x.append(img_np[:,:,i])
            test_y.append(msk_np[:,:,i])
            test_x.append(img_np[:,i,:])
            test_y.append(msk_np[:,i,:])
            test_x.append(img_np[i,:,:])
            test_y.append(msk_np[i,:,:])
    
    tensor_x = Tensor.fromarray(test_x)
    tensor_y = Tensor.fromarray(test_y)

    tensor_x.save("miccai_tsX_preprocessed.bin", "bin")
    tensor_y.save("miccai_tsY_preprocessed.bin", "bin")

    # Move binary files to dataset folder
    os.rename("miccai_trX_preprocessed.bin", MICCAI_PATH+"bin/miccai_trX_preprocessed.bin")
    os.rename("miccai_trY_preprocessed.bin", MICCAI_PATH+"bin/miccai_trY_preprocessed.bin")
    os.rename("miccai_tsX_preprocessed.bin", MICCAI_PATH+"bin/miccai_tsX_preprocessed.bin")
    os.rename("miccai_tsY_preprocessed.bin", MICCAI_PATH+"bin/miccai_tsY_preprocessed.bin")


if __name__ == "__main__":
    if MICCAI_PATH == None:
        print("ERROR - Path to DRIVE dataset folder is not set")
    else:
        order = input("WARNING - Data preprocessing must be executed only once. Proceed? (y/n)")
        if order == "y":
            main()
        else:
            print("INFO - Execution aborted")

