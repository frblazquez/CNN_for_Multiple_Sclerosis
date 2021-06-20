import numpy as np
from scipy import ndimage

def DSC(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def vol_dif(im1, im2):
    """
    absolute difference in volume 
    """
    
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.abs(im2.sum() - im1.sum()) / im1.sum()

def TP(my_seg, labels):
    return np.logical_and(my_seg==1,labels==1)

def FP(my_seg, labels):
    return np.logical_and(my_seg==1,labels==0)

def FN(my_seg, labels):
    return np.logical_and(my_seg==0,labels==1)

def TPR(my_seg, labels):
    TP_sum = np.sum(TP(my_seg, labels))
    FN_sum = np.sum(FN(my_seg, labels))
    return TP_sum / (TP_sum + FN_sum)

def PPV(my_seg, labels):
    TP_sum = np.sum(TP(my_seg, labels))
    FP_sum = np.sum(FP(my_seg, labels))
    return TP_sum / (TP_sum + FP_sum)

    

