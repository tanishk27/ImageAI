import os
import cv2
import io
import glob
import math
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage.morphology import binary_erosion
from pymatting.util.util import stack_images
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
import numpy as np
from skimage import io, transform
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL
import torch
torch.cuda.is_available = lambda : False
from numpy import asarray
import pandas as pd
import warnings
import subprocess
import imageio as iio

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from scripts.data_loader import RescaleT
from scripts.data_loader import ToTensor
from scripts.data_loader import ToTensorLab
from scripts.data_loader import SalObjDataset
import skimage.exposure
import signal
from subprocess import Popen, PIPE, TimeoutExpired
from time import monotonic as timer
# def sig_handler(signum, frame):
#     #deal with the signal.. 
#     print("segfault error !!")
# signal.signal(signal.SIGSEGV, sig_handler)
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'_mask.png')

def model_mask(net, Shell_File, PARENT_FOLDER, img_name_list, yolo_model_path, UPLOAD_FOLDER):
    #img_name_list = glob.glob(UPLOAD_FOLDER + os.sep + '*')
    # print("Hi")
    n = len(img_name_list)
    extracted_list = []
    psd_list = []

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(720),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    for i_test, data_test in enumerate(test_salobj_dataloader):
        # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
        # print(d1)
        # i = 0
        # normalization
        # while i == i_test:
        pred = d1[0,:,:,:]
        # print("*************************  ", d1.shape)
        pred = normPRED(pred)
        # save results to test_results folder
        if not os.path.exists(PARENT_FOLDER):
            os.makedirs(PARENT_FOLDER, exist_ok=True)
        save_output(img_name_list[i_test],pred,PARENT_FOLDER)

        del d1,d2,d3,d4,d5,d6,d7
        # print("$$$$$$",img_name_list[i_test])
    
        psd, save_path = post_process(Shell_File, img_name_list[i_test],yolo_model_path, PARENT_FOLDER, UPLOAD_FOLDER)
        if os.path.exists(save_path) :
            extracted_list.append(save_path)
        if os.path.exists(psd) :
            psd_list.append(psd)
    return psd_list, extracted_list

def estimate_trimap(mask) :
    #mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)

    # Create empty trimap
    trimap = np.zeros_like(mask_array)

    # Set pixels in the trimap to be 0 (background), 128 (unknown), or 255 (foreground) based on the mask
    # trimap[mask_array == 0] = 0
    # trimap[(mask_array > 0) & (mask_array < 255)] = 128
    # trimap[mask_array == 255] = 255
    foreground_threshold = 253
    background_threshold = 17

    # Set the foreground pixels in the trimap
    trimap[mask_array >= foreground_threshold] = 255

    # Set the background pixels in the trimap
    trimap[mask_array <= background_threshold] = 0

    # Set the unknown pixels in the trimap
    trimap[(mask_array > background_threshold) & (mask_array < foreground_threshold)] = 128
    return trimap

def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    # img.thumbnail((2456, 3680), Image.LANCZOS)

    mask = mask.resize(img.size, Image.NEAREST)
    # mask.save('mask.png')

    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

    is_foreground = binary_erosion(is_foreground, structure=structure, border_value=1)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = estimate_trimap(mask)
    # trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    # trimap[is_foreground] = 255
    # trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)
    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.CUBIC)
    # cutout = cutout.filter(ImageFilter.EDGE_ENHANCE) 
    return cutout

def sharpennet(area):     
    for k in range(0,50):
        # area[:,:] = np.where(area[:,:]* 1.02 < 255, (area[:,:] * 1.02).astype(np.uint8) , area[:,:])
        area[:,:] = np.where(area[:,:]* 1.02 < 255, (area[:,:] * 1.02).astype(np.uint8) , area[:,:])
    try :
        area2 = cv2.threshold(area, 127, 255, cv2.THRESH_BINARY)[1]
        area2 = cv2.GaussianBlur(area2, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
        area2 = skimage.exposure.rescale_intensity(area2, in_range=(127,255), out_range=(0,255))
        return area2
    except :
        return area

def sharpen(area):
    try :
        area2 = cv2.threshold(area, 127, 255, cv2.THRESH_BINARY)[1]
        area2 = cv2.GaussianBlur(area2, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
        area2 = skimage.exposure.rescale_intensity(area2, in_range=(127,255), out_range=(0,255))
        return area2
    except :
        return area

def optimize_face(img):
    try :
        mask = img
        # Apply Gaussian blur
        # blur = cv2.GaussianBlur(img, (3, 3), 0)

        # # Apply Laplacian edge detection
        # laplacian = cv2.Laplacian(blur, cv2.CV_64F)

        # # Normalize the edges
        # edges = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC4)

        # sharpened = cv2.addWeighted(img, 1.5, edges, -0.9, 0)
        # # sharpened = cv2.dilate(sharpened, kernel, iterations=None)
        # kernel_size = 3
        # sigma = 1

        # mask = cv2.GaussianBlur(sharpened, (kernel_size, kernel_size), sigma)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        # mask = cv2.erode(mask, kernel, iterations=None)
        return mask
    except :
        return img

def optimize_mid(img):
    try :
        blur = cv2.GaussianBlur(img, (3, 3), 0)

        # Apply Laplacian edge detection
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)

        # Normalize the edges
        edges = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC4)
        sharpened = cv2.addWeighted(img, 1.8, edges, -0.4, 0)
        
        kernel_size = 3
        sigma = 1

        mask = cv2.GaussianBlur(sharpened, (kernel_size, kernel_size), sigma)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        mask = cv2.erode(mask, kernel, iterations=None)
        return mask
    except :
        return img

def optimize_body(img):
    try :
        gray = img
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply Laplacian edge detection
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)

        # Normalize the edges
        edges = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        sharpened = cv2.addWeighted(img, 2.7, edges, -0.5, 0)

        kernel_size = 3
        sigma = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
        mask_smoothed = cv2.GaussianBlur(sharpened, (kernel_size, kernel_size), sigma)
        mask = cv2.erode(mask_smoothed, kernel, iterations=None)
        return mask
    except :
        return img

def optimize_full(img):
    try :
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

        # Apply the morphological erosion operation to the mask using the structuring element
        mask_eroded = cv2.erode(img, se, iterations=None)
        cv2.imwrite('tmp/eroded.png', mask_eroded)
        
        # Subtract the eroded mask from the original mask to obtain the mask without edge pixels
        mask_without_edges = cv2.subtract(img, mask_eroded)
        mask_without_edges = mask_without_edges * 0.7
        
        #mask_without_edges = mask_without_edges * 0.7
        cv2.imwrite('tmp/edges.png', mask_without_edges)
        edges = cv2.imread('tmp/edges.png', cv2.IMREAD_GRAYSCALE)
        eroded = cv2.imread('tmp/eroded.png', cv2.IMREAD_GRAYSCALE)
        mask_with_intensity = cv2.addWeighted(eroded, 1.2, edges, 0.7, 0)
        
        # Save the edges to an array
        return mask_with_intensity
    except :
        return img


def fit_50(mid, x_, w_, h_, h_right1, h_left1, code):
    try :
        if code == 1 :
            h_right1 = 0
            rf1 = mid[:h_-h_right1, x_:math.ceil(x_+w_/2)]
            lf1 = mid[:h_-h_left1, math.ceil(x_+w_/2):x_+w_]
            rfb1 = mid[h_-h_right1:, x_:math.ceil(x_+w_/2)]
            lfb1 = mid[h_-h_left1:, math.ceil(x_+w_/2):x_+w_]

            mid[:h_-h_right1, x_:math.ceil(x_+w_/2)] = optimize_face(rf1)
            mid[:h_-h_left1, math.ceil(x_+w_/2):x_+w_] = optimize_face(lf1)
            mid[h_-h_right1:, x_:math.ceil(x_+w_/2)] = optimize_mid(rfb1)
            mid[h_-h_left1:, math.ceil(x_+w_/2):x_+w_] = optimize_mid(lfb1) 
            print('done')
        if code == 2:
            h_left1 = 0
            rf1 = mid[:h_-h_right1, x_:math.ceil(x_+w_/2)]
            lf1 = mid[:h_-h_left1, math.ceil(x_+w_/2):x_+w_]
            rfb1 = mid[h_-h_right1:, x_:math.ceil(x_+w_/2)]
            lfb1 = mid[h_-h_left1:, math.ceil(x_+w_/2):x_+w_]

            mid[:h_-h_right1, x_:math.ceil(x_+w_/2)] = optimize_face(rf1)
            mid[:h_-h_left1, math.ceil(x_+w_/2):x_+w_] = optimize_face(lf1)
            mid[h_-h_right1:, x_:math.ceil(x_+w_/2)] = optimize_mid(rfb1)
            mid[h_-h_left1:, math.ceil(x_+w_/2):x_+w_] = optimize_mid(lfb1) 
            print('done')
        if code == 3 :
            pass
        
        
    except :
        print("exception occured in fit50")
        pass

def fit_25(mid, x, y, w, h_right, h_left, code):
    try :
        if code == 0:
            rf = mid[:h_right, x:math.ceil(x+w/2)]
            lf = mid[:h_left, math.ceil(x+w/2):x+w]
            rfb = mid[h_right:, x:math.ceil(x+w/2)]
            lfb = mid[h_left:, math.ceil(x+w/2):x+w]

            mid[:h_right, x:math.ceil(x+w/2)] = optimize_face(rf)
            mid[:h_left, math.ceil(x+w/2):x+w] = optimize_face(lf)
            mid[h_right:, x:math.ceil(x+w/2)] = optimize_body(rfb)
            mid[h_left:, math.ceil(x+w/2):x+w] = optimize_body(lfb)
        if code == 1 :
            rf = mid[:, x:math.ceil(x+w/2)]
            lf = mid[:h_left, math.ceil(x+w/2):x+w]
            lfb = mid[h_left:, math.ceil(x+w/2):x+w]

            mid[:h_left, math.ceil(x+w/2):x+w] = optimize_face(lf)
            mid[:, x:math.ceil(x+w/2)] = optimize_mid(rf)
            mid[h_left:, math.ceil(x+w/2):x+w] = optimize_body(lfb)
        if code == 2 :
            rf = mid[:h_right, x:math.ceil(x+w/2)]
            lf = mid[:, math.ceil(x+w/2):x+w]
            rfb = mid[h_right:, x:math.ceil(x+w/2)]
            
            mid[:h_right, x:math.ceil(x+w/2)] = optimize_face(rf)
            mid[:, math.ceil(x+w/2):x+w] = optimize_mid(lf)
            mid[h_right:, x:math.ceil(x+w/2)] = optimize_body(rfb)
        print("CODE ", code)
    except :
        print('exception occured in fit50')
        pass

def detect(p, image_path, mask_path, save_path, model):
    xmax = []
    xmin = []
    ymin = []
    ymax = []
    hlist = []
    wlist = []
    conf =[]
    factor_x = 50
    factor_y = 100
    image = cv2.imread(image_path)
    if not p.empty: 
        n = len(p.xmin)
        print("%%%%",n)   
        for i in range(0,n):
            print("Confidence : ", p.confidence[i])
            conf.append(p.confidence[i])
            xmin.append(int(p.xmin[i]))
            ymin.append(int(p.ymin[i]))
            xmax.append(int(p.xmax[i]))
            ymax.append(int(p.ymax[i]))
            wlist.append(int(p.xmax[i] - p.xmin[i])+ factor_x) 
            hlist.append(int(p.ymax[i] - p.ymin[i])+ factor_y)
            face = image[ymin[i]:(ymin[i]+hlist[i]), xmin[i]:(xmin[i]+wlist[i])]
            face_right = image[ymin[i]:ymin[i]+hlist[i], xmin[i]:math.ceil(xmin[i]+(wlist[i]/2))]
            face_left = image[ymin[i]:ymin[i]+hlist[i], math.ceil(xmin[i]+(wlist[i]/2)):xmin[i]+wlist[i]]
            cv2.imwrite(f'tmp/{i}.png', face)
            cv2.imwrite(f'tmp/{i}{i}.png', face_right)
            cv2.imwrite(f'tmp/{i}{i}{i}.png', face_left)
        
        # print(xmax, xmin, ymax, ymin)
        w = max(xmax) - min(xmin) + factor_x
        h = max(ymax) - min(ymin) + factor_y
        x = min(xmin)
        y = min(ymin)
        # print("actual", h)

        img_ = cv2.imread(mask_path)

        width = img_.shape[1] 
        down = img_[y+h:, 0:]
        up = img_[:y, :]
        mid = img_[y:y+h, :]
        rc = mid[:, :x]
        lc = mid[:, x+w:]  

        result_right = model('tmp/00.png')
        result_left = model('tmp/000.png')   

        p_right = result_right.pandas().xyxy[0]
        p_left = result_left.pandas().xyxy[0]     

        if p_right.empty and p_left.empty :
            face = img_[y:y+h, x:x+w]
            img_[y:y+h, x:x+w] = optimize_mid(face) 
        else :
            if p_right.empty :
                code = 1
            if p_left.empty :
                code = 2
            if (not p_right.empty) and (not p_left.empty) :
                code = 0
            try :
                ymin_right = p_right.ymin[0]
                ymax_right = p_right.ymax[0]
                h_right = math.floor(ymax_right - ymin_right)
                # print("right ", h_right)
            except :
                print('exception')
                h_right = h
            try :
                ymin_left = p_left.ymin[0]
                ymax_left = p_left.ymax[0]
                h_left = math.floor(ymax_left - ymin_left)
                # print("left ", h_left)
            except :
                print('exception')
                h_left = h 
            
            fit_25(mid, x, y, w, h_right, h_left, code)

        if n > 1 :
            if n == 2: 
                print("N = 2....\n")
                result_right1 = model('tmp/11.png')
                result_left1 = model('tmp/111.png')
                # print(h)
                # print("HEIGHT", height)

                p_right1 = result_right1.pandas().xyxy[0]
                p_left1 = result_left1.pandas().xyxy[0]

                if p_right1.empty :
                    code = 1
                elif p_left1.empty :
                    code = 2
                elif (not p_right1.empty) and (not p_left1.empty) :
                    code = 0
                else :
                    code = 3

                print("CODE : ", code)
                try :
                    print("In try 1")
                    ymin_right1 = p_right1.ymin[0]
                    ymax_right1 = p_right1.ymax[0]
                    h_right1 = math.floor(ymax_right1 - ymin_right1)
                    print("right ", h_right1)
                except :
                    print("Exception occured in first")
                    ##apply counter
                    h_right1 = h
                    
                try :
                    print("In try 2")
                    ymin_left1 = p_left1.ymin[0]
                    ymax_left1 = p_left1.ymax[0]
                    h_left1 = math.floor(ymax_left1 - ymin_left1)
                    print("left ", h_left1)
                except :
                    print("Exception occured in second")
                    h_left1 = h
            
                minimumpos = xmax.index(min(xmax))
                maximumpos = xmin.index(max(xmin))
                
                if (p.confidence[0] < p.confidence[1]) :
                    pos = 0
                else :
                    pos = 1
                w_ = wlist[pos]
                h_ = hlist[pos]
                x_ = xmin[pos]
                y_ = ymin[pos]
                
                # print('h_ : ', h_)
                fit_50(mid, x_, w_, h_, h_right1, h_left1, code)

                middlest = mid[:, xmax[minimumpos]:xmin[maximumpos]]
                if xmax[minimumpos] - xmin[maximumpos] < 0 :
                    mid[:, xmax[minimumpos]:xmin[maximumpos]] = optimize_body(middlest)
                    #mid[:height-+h_right1, xmin[maximumpos]+]

                minpos = ymin.index(min(ymin))
                maxpos = ymax.index(max(ymax))
                if xmin[minpos] - xmax[minpos] > 0 :
                    middlestrc = mid[hlist[minpos]:, xmin[minpos]:xmax[minpos]]
                    mid[hlist[minpos]:, xmin[minpos]:xmax[minpos]] = optimize_body(middlestrc)
                    
                if xmin[maxpos] - xmax[maxpos] > 0 :
                    middlestlc = mid[:(h - hlist[maxpos]), xmin[maxpos]:xmax[maxpos]]
                    mid[:(h - hlist[maxpos]), xmin[maxpos]:xmax[maxpos]] = optimize_body(middlestlc)
            
            else :
                pos = conf.index(max(conf))
                print(pos)
                for i in range (0, n):
                    if i == pos :
                        continue
                    else :
                        result_right1 = model(f'tmp/{i}{i}.png')
                        result_left1 = model(f'tmp/{i}{i}{i}.png')

                        p_right1 = result_right1.pandas().xyxy[0]
                        p_left1 = result_left1.pandas().xyxy[0]

                        if p_right1.empty :
                            code = 1
                        if p_left1.empty :
                            code = 2
                        if (not p_right1.empty) and (not p_left1.empty) :
                            code = 0

                        w_ = wlist[i]
                        h_ = hlist[i]
                        x_ = xmin[i]
                        y_ = ymin[i]

                        try :
                            ymin_right1 = p_right1.ymin[0]
                            ymax_right1 = p_right1.ymax[0]
                            h_right1 = math.floor(ymax_right1 - ymin_right1)
                            print("right ", h_right1)
                        except :
                            h_right1 = h_

                        try :
                            ymin_left1 = p_left1.ymin[0]
                            ymax_left1 = p_left1.ymax[0]
                            h_left1 = math.floor(ymax_left1 - ymin_left1)
                            print("left ", h_left1)      
                        except :
                            h_left1 = h_                 
                        
                        fit_50(mid, x_, w_, h_, h_right1, h_left1, code)
                    

                # height = max(ymax) - min(ymin)
                minpos = ymin.index(min(ymin))
                maxpos = ymax.index(max(ymax))
                middlestrc = mid[hlist[minpos]:, xmin[minpos]:xmax[minpos]]
                middlestlc = mid[:(h - hlist[maxpos]), xmin[maxpos]:xmax[maxpos]]
                if  xmin[minpos] - xmax[minpos] > 0 :
                    mid[hlist[minpos]:, xmin[minpos]:xmax[minpos]] = optimize_body(middlestrc)
                if xmin[maxpos] - xmax[maxpos] > 0 :    
                    mid[:(h - hlist[maxpos]), xmin[maxpos]:xmax[maxpos]] = optimize_body(middlestlc)
        
        up = optimize_body(up)
        mid[:,:x] = optimize_body(rc)
        mid[:,x+w:] = optimize_body(lc)
        down = optimize_body(down)

        for i in range(0,n):
            if p.iloc[i, 5] == 1 :
                print("found net")
                net = mid[:, xmin[i]:xmax[i]]
                mid[:, xmin[i]:xmax[i]] = optimize_face(net)

        # down[y:y+h, x:x+w] = head_copy[:,:]
        # final_mask=down
        final_mask = np.concatenate((up, mid, down), axis=0)
        final_mask = np.array(final_mask)
        final_mask = cv2.medianBlur(final_mask, 3)
        cv2.imwrite(mask_path, final_mask)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.imread(image_path)
        
        rgba = cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2], mask))
        cv2.imwrite(save_path, rgba)
        
           
    if p.empty:
        print('Hairs not found !!')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        final_mask = optimize_body(mask)
        
        image = cv2.imread(image_path)
        
        rgba = cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2], final_mask))
        cv2.imwrite(save_path, rgba)

def PSD_Generation(Shell_File, Raw_Image, Extracted_PNG, PARENT_FOLDER):	
    #filename = "output.psd"
    filename_base = os.path.basename(Raw_Image)
    filename_, file_extension = os.path.splitext(filename_base)
    file = filename_+'.psd'
    Output_PSD = os.path.join(PARENT_FOLDER, file)
    print("Starting PSD Generation ....\n")

    start = timer()
    with Popen([Shell_File, '-a', Raw_Image, '-b', Extracted_PNG, '-c', Output_PSD], stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid) as process:
        try:
            output = process.communicate(timeout=25)[0]
        except TimeoutExpired:
            os.killpg(process.pid, signal.SIGINT) # send signal to the process group
            output = process.communicate()[0]
            print('Timeout Expired')
        finally :
            print('Elapsed seconds: {:.2f}'.format(timer() - start))
            return Output_PSD
    
def post_process(Shell_File, image_path,yolo_model_path, PARENT_FOLDER, UPLOAD_FOLDER):
    # for image_path in image_list:
    if not os.path.exists(image_path):
        print('Cannot find input path: {0}'.format(image_path))
    else :
        filename_base = os.path.basename(image_path)
        filename_, file_extension = os.path.splitext(filename_base)
        file = filename_+'_mask.png'
        mask_path = os.path.join(PARENT_FOLDER, file)
        # image = Image.open(image_path).convert("RGB")
        
        image = cv2.imread(image_path)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # size = os.stat(image_path).st_size
        file = filename_ + '.png'
        image_path = os.path.join(UPLOAD_FOLDER, file)
        image.save(image_path)
        
        mask = Image.open(mask_path).convert("L")
        alpha_matting_foreground_threshold = 240
        alpha_matting_background_threshold = 20
        alpha_matting_base_size = 3000
        alpha_matting_erode_structure_size = 1
        size = os.stat(image_path).st_size
        if size < 3096150 :
            if image.height > image.width :
                if image.width < 3000 :
                    alpha_matting_base_size = int(image.width)
                    alpha_matting_erode_structure_size = 1
                else :
                    alpha_matting_base_size = 3000
                    alpha_matting_erode_structure_size = 1

            elif image.height < image.width :
                if image.height < 2900 :
                    alpha_matting_base_size = int(image.height)	
                    alpha_matting_erode_structure_size = 1
                else :
                    alpha_matting_base_size = 3000
                    alpha_matting_erode_structure_size = 1
        if size > 30000000 :  
            alpha_matting_foreground_threshold = 250 
            alpha_matting_base_size = 3840   
        model = yolo_model_path
        # model = torch.hub.load('.', 'custom', path=yolo_model_path, source='local')
        results = model(image_path)
        # labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        #results.print()
        p = results.pandas().xyxy[0]
        n = len(p.xmin)
        for i in range(0,n):
            if p.iloc[i, 5] == 1 :
                alpha_matting_erode_structure_size = 1
        cutout = alpha_matting_cutout(
                    image,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_structure_size,
                    alpha_matting_base_size,
                )
        print(alpha_matting_erode_structure_size, alpha_matting_base_size)
        save_path = filename_ + '.png'
        save_path = os.path.join(PARENT_FOLDER, save_path)
        print(save_path)
        print(image_path)
        # cutout.save('cutout.png', icc_profile=cutout.info.get('icc_profile'))
        mask = np.array(cutout)
        mask = mask[:, :, 3]
        cv2.imwrite(mask_path, mask)
        # mask = iio.imread(mask_path)
        # mask = optimize_full(mask)
        # image = iio.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rgba = cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2], mask))
        detect(p, image_path, mask_path, save_path, model)
        try :
            psd = PSD_Generation(Shell_File, image_path, save_path, PARENT_FOLDER)
            return psd, save_path
        except :
            print('failed to load psd !!')
            psd = " "
            return psd, save_path

def predict(net, Shell_File, image_list,yolo_model_path, PARENT_FOLDER, UPLOAD_FOLDER):
    psd_list, extracted_list = model_mask(net, Shell_File, PARENT_FOLDER, image_list, yolo_model_path, UPLOAD_FOLDER)
    # save_path = post_process(image_list,yolo_model_path, PARENT_FOLDER)
    return psd_list, extracted_list
