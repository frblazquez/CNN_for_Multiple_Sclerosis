# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
#
# École polytechnique fédérale de Lausanne, Switzerland
# Deephealth project
#
# Description:
# Dice metrics for the validation imgages computation

import numpy as np
from PIL import Image


IMAGES_IDXS = [175, 943, 1426]
NUM_EPOCHS = 50
THRESHOLD = 0.65

for img_idx in IMAGES_IDXS:
    mask_np = (np.array(Image.open("tr"+str(img_idx)+"/tr"+str(img_idx)+"_mask.jpg"))/255).astype(bool)
    for i in range(NUM_EPOCHS):
        img_np = np.array(Image.open("tr"+str(img_idx)+"/tr"+str(img_idx)+"_e"+str(i)+"_pred.jpg"))[:,:,0] > THRESHOLD
        print(2* np.sum(np.logical_and(img_np, mask_np)) / (np.sum(mask_np) + np.sum(img_np)))

