from scripts.psdgen import PSD_Generation
from scripts.rawtoexta import predict

import numpy as np
from PIL import Image
import os

def Final_PSD(net, Shell_File, Image_List, YOLO_Model_path, PARENT_FOLDER, UPLOAD_FOLDER, RescaleVal):
	extracted_list = predict(net, Image_List, YOLO_Model_path, PARENT_FOLDER, UPLOAD_FOLDER, RescaleVal)
	output = list(zip(Image_List, extracted_list))
	
	psd_list = []
	for image in output:
		if (os.path.splitext(os.path.basename(image[0]))[0] == os.path.splitext(os.path.basename(image[1]))[0]):
			psd = PSD_Generation(Shell_File, image[0], image[1], PARENT_FOLDER)
			psd_list.append(psd)
		else :
			print('ERROR : Extracted Image for file is not generated !!')
			continue
	return psd_list, extracted_list
	
