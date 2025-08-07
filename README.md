# Image AI : An Extraction-Tool

**Step 1 :**
 + Place GIMP folder in computer/home/user/home/.config/
    + Note : Insure that there is lyers-to-psd.scm file is there inside GIMP/2.10/scripts/

**Step 2 :**
 + Give read/write permission to all folders in directory using :
    + chmod a+x folder_name

**Step 3 :**
 + Install requirements by creating virtual env of python3.7

**Step 4 :**
 + Run the Flask API using command :
    + python main.py

### Project Structure :
**Flask API :**
+ Scripts :
	- rawtoext.py : Predicts foreground of raw image -> generate GT-> Postprocess GT -> genaerate extracted PNG (Inculdes u2net primary GT prediction, yolo face detection, alpha matting, sharpening of body edges)
	- dataloader.py : loads data for u2net model to predict GT
	- psd.sh : Shell file which runs scrit-fu program file layers-to-psd.scm of GIMP, returns linked PSD file with GT and extracted PNG to psdgen
	- psdgen.py : Runs shell file psd.sh and returns PSD file
	- predict.py : returns list of PSD's and Extracted PNG's
+ Templates :
	- Frontend Files(HTML, CSS, JS) of FLASK API 
+ Static :
	- uploads : (Source) Contains Uploaded raw images
	- results : (Destination) Contains resultant output files - PSD's, PNG's, GT's
+ Pretrained :
	- u2net model
	- yolo v5 model

### Training :

**1. Yolo V5 For hair detection :**
 __https://github.com/ultralytics/yolov5__
 
**2. U2NET for foreground extraction :**
__https://github.com/xuebinqin/U-2-Net__

__Training script for u2net training :__

```python
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP
torch.cuda.empty_cache()

bce_loss = nn.BCELoss(size_average=True)
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))
	return loss0, loss

model_name = 'u2netp' #'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = os.path.join('DUTS', 'badminton', 'im_aug' + os.sep)
tra_label_dir = os.path.join('DUTS', 'badminton', 'gt_aug' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 250
batch_size_train = 2
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]
	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]
	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(512),
        # RandomCrop(480),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)
net.load_state_dict(torch.load('saved_models/u2netp/best/u2netp_2812.pth'))
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0
running_tar_loss = 0
ite_num4val = 0
save_frq = 2253 # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1
        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        # y zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        loss.backward()
        optimizer.step()
        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
```

### Dataset Preparation :

**1. Resize images :**

```python
from PIL import Image
import os
import cv2

in_path = "path/To/Source/Folder/"
out_path = "path/To/Destination/Folder"

for f in os.listdir(in_path):
	f1 = os.path.join(in_path, f)
	print(f1)
	img=Image.open(f1)
	print(img.size)
	img = img.resize((int(img.size[0]*5), int(img.size[1]*5)))
	print(img.size)
	f2 = os.path.join(out_path, f)
	img.save(f2)
```
**2. Create Ground Truth(GT) from Extracted PNG's :**

```python
from PIL import Image
import os
import cv2

in_path = "path/To/Source/Folder/"
out_path = "path/To/Destination/Folder"

for f in os.listdir(in_path):
	f = os.path.join(in_path, f)
	print(f)
	img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
	img_gray = img[:, :, 3]
	f2 = os.path.join(out_path, f)
	cv2.imwrite(f2, img_gray)
```
**3. Find biggest contour from GT :**
```python
def grab_rgb(image, c):
    pixels=[]
    mask = image
    points = np.where(mask == 0)

    for point in points:
        pixel = (mask[point[1], point[0]])
        pixel = pixel.tolist()
        pixels.append(pixel)

    pixels = [tuple(l) for l in pixels]
    car_color = (pixels[1])
    r = car_color[0]
    g = car_color[1]
    b = car_color[2]
    pixel_string = r,g,b
    return pixel_string

def countlen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9, 9), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    thresh = cv2.threshold(gray, 127, 255, 0)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cntrs = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cntrs1 = cntrs

    cntrs = (cntrs1[0] if len(cntrs1) == 2 else cntrs1[1])

    if len(cntrs) > 1:
        counter = 0
        while counter < len(cntrs) + 1:
            areal = []
            n = 0
            m = 0
            for c in cntrs:
                p = grab_rgb(img, c)
                if p == (0, 0, 0):
                    area = cv2.contourArea(c)
                    areal.append(area)
                    m += 1
                    area_thresh = max(areal)
                    if area == area_thresh:
                        big_contour = c
                        big_contour = np.array(big_contour)
                        for i in big_contour:
                            if c.all() == i.all() and m != 0:
                                new = np.delete(cntrs, n)
                    n += 1
            counter += 1
        cnts = new
        i = 0
        for c in cnts:
            if i <= len(cnts):
                cv2.drawContours(img, cnts, i, (0, 0, 0), -1)
            i += 1
    return img
 countlen(grayscale_image)
 ```

**4. Generate trimap from GT :**
```python
from PIL import Image
import sys
import numpy as np
import cv2
import math
import scipy
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
import sys
import numpy as np
import numpy as np
from numpy import asarray
from PIL import Image, ImageOps
import cv2
import os
import numpy
import sys
def generate_trimap(mask_path,eroision_iter=8,dilate_iter=6):
    mask =  mask_path
    mask = cv2.imread(mask,0)
    mask[mask==1] = 255
    d_kernel = np.ones((3,3))
    erode  = cv2.erode(mask,d_kernel,iterations=eroision_iter)
    dilate = cv2.dilate(mask,d_kernel,iterations=dilate_iter)
    unknown1 = cv2.bitwise_xor(erode,mask)
    unknown2 = cv2.bitwise_xor(dilate,mask)
    unknowns = cv2.add(unknown1,unknown2)
    unknowns[unknowns==255]=127
    trimap = cv2.add(mask,unknowns)
    #cv2.imwrite("dilate_xor+erode_xor.png",unknowns)
    #cv2.imwrite("erode.png",erode)
    cv2.imwrite("tmp/SB5_9819_trimap.png",trimap)
    print(trimap.shape)
    labels = trimap.copy()
    #labels = labels.tolist()
    labels = labels.astype(float)
    labels[trimap==127] = 0.5 #unknown
    labels[trimap==255] = 1.0 #foreground
    labels[trimap==0] = 0.0 #background
    #print(labels)
    #print(labels.ndim)
    return labels

trimap_ = generate_trimap('tmp/SB5_9819_mask.png')

```
**5. Remove Extracted PNG's whoes Raw images are not present :**
```python
import os
import sys
import shutil
folder_raw = 'raw2/'
folder_extracted = '/home/infogen/220428_14450/alpha2/'
dest = './unseen_'

lst = []
for raw_path in os.listdir(folder_raw) :
    e_path, e_ext = os.path.splitext(raw_path)
    e_path = e_path+'.png'
    lst.append(e_path)
    
for path in os.listdir(folder_extracted) :
    if not path in lst:
        ext = folder_extracted+path
        #print(ext)
        os.remove(ext)
        print("Removed -", ext)
```
**6. Remove Raw images whoes Extracted PNG's are not present :**
```python
lst = []
for extracted_path in os.listdir(folder_extracted) :
    e_path, e_ext = os.path.splitext(extracted_path)
    e_path = e_path+'.jpg'
    lst.append(e_path)
    
for path in os.listdir(folder_raw) :
    if not path in lst:
        raw = folder_raw+path
        #print(raw)
        dst_path = dest+path
        shutil.copy(raw, dst_path)
        os.remove(raw)
        print("Removed -", raw)
```
**7. Generalize Raw image :**
```python
def get_image(image_path):
    rgb = Image.open(image_path)
    rgb.load()
    data = np.asarray(rgb)
    data = data.astype('float32')
    data /= 255.0
    return data
```
### References :
+ __https://github.com/ZHKKKe/MODNet__
+ __https://github.com/danielgatis/rembg__
+ __https://purephotos.app/__
+ __https://pixspy.com/__
+ __https://www.analyticsvidhya.com/blog/2021/08/sharpening-an-image-using-opencv-library-in-python/__
+ __https://github.com/nadermx/backgroundremover__
