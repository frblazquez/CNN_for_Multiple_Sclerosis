import numpy   as np
import nibabel as nib
import skimage.transform as skTrans
import random

MICCAI_PATH = '/scrap/users/blazquez/datasets/miccai2016/'

validation1_idx = 2
validation2_idx = 10
validation3_idx = 13

test_idxs = [validation1_idx,validation2_idx,validation3_idx]
train_idxs = [i for i in range(15) if i not in test_idxs]

train_x = np.empty((0,256,256,1))
train_y = np.empty((0,256,256,1))
test_x  = np.empty((0,256,256,1))
test_y  = np.empty((0,256,256,1))


for i in test_idxs:
    img_name = MICCAI_PATH+'preprocessed/s'+str(i+1)+'/FLAIR_preprocessed.nii'
    msk_name = MICCAI_PATH+'unprocessed/s'+str(i+1)+'/Consensus.nii'
    img_np = nib.load(img_name).get_fdata()
    msk_np = nib.load(msk_name).get_fdata()
    
    # Images resizing!
    img_np = skTrans.resize(img_np, (256,256,256), order=1, preserve_range=True)
    msk_np = skTrans.resize(msk_np, (256,256,256), order=1, preserve_range=True)
    
    for j in range(256):
        test_x = np.append(test_x, np.expand_dims(np.expand_dims(img_np[:,:,j],axis=0), axis=3), axis=0)
        test_y = np.append(test_y, np.expand_dims(np.expand_dims(msk_np[:,:,j],axis=0), axis=3), axis=0)


print(test_x.shape)
print(test_y.shape)

np.save(MICCAI_PATH+"npy/miccai2016_tsx", test_x)
np.save(MICCAI_PATH+"npy/miccai2016_tsy", test_y)

for i in train_idxs:
    img_name = MICCAI_PATH+'preprocessed/s'+str(i+1)+'/FLAIR_preprocessed.nii'
    msk_name = MICCAI_PATH+'unprocessed/s'+str(i+1)+'/Consensus.nii'
    img_np = nib.load(img_name).get_fdata()
    msk_np = nib.load(msk_name).get_fdata()
    
    # Images resizing!
    img_np = skTrans.resize(img_np, (256,256,256), order=1, preserve_range=True)
    msk_np = skTrans.resize(msk_np, (256,256,256), order=1, preserve_range=True)
    
    # Do whatever you want with the image resized as numpy array
    for j in range(256):
        train_x = np.append(train_x, np.expand_dims(np.expand_dims(img_np[:,:,j],axis=0), axis=3), axis=0)
        train_y = np.append(train_y, np.expand_dims(np.expand_dims(msk_np[:,:,j],axis=0), axis=3), axis=0) 


print(train_x.shape)
print(train_y.shape)

np.save(MICCAI_PATH+"npy/miccai2016_trx", train_x)
np.save(MICCAI_PATH+"npy/miccai2016_try", train_y)

print("INFO - Execution finished")


