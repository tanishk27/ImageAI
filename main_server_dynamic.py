import resource
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
# from scripts.predict import Final_PSD
from scripts.rawtoexta import predict
from PIL import Image
import time
import numpy as np
import os
from glob import glob
# from zipfile import ZipFile
import torch
import ftputil
import pathlib
import requests
from scripts.model.u2net import U2NET
#from scripts.model.u2net import U2NETP
UPLOAD_FOLDER = 'static/uploads/'
PARENT_FOLDER = "static/results/"

#Model_Path = "pretrained/new5_2104.pth"
Model_Path = "pretrained/new4_1704.pth"

Shell_File = "scripts/psd.sh"
#YOLO_Model_path = "pretrained/hair_0601.pt"
YOLO_Model_path = "pretrained/best_only_hair_500_myg_latest.pt"
YOLO_Model_path = torch.hub.load('.', 'custom', path=YOLO_Model_path, source='local')
net = U2NET(3,1)
net.load_state_dict(torch.load(Model_Path, map_location='cpu'))
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PARENT_FOLDER'] = PARENT_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 32 * 2048 * 2048
ALLOWED_EXTENSIONS = ('.jpg', ".jpeg", ".JPG", "JPEG")

import faulthandler
faulthandler.enable()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def divide_chunks(l, n):
    # looping till length l
	for i in range(0, len(l), n):
		yield l[i:i + n]     
		
@app.route('/')
def upload_form():
	#flash('Image successfully uploaded !!')
	return render_template('input.html')

def download(ftp, folder, out, errorlist):
	for item in ftp.walk(folder):
		print("Creating dir " + item[0])
		p = pathlib.Path(item[0])
		path2 = pathlib.Path(out).joinpath(*p.parts[1:])
		try :
			# os.mkdir(path2)
			ftp.mkdir(path2)
		except ftputil.error.PermanentError :
			pass

		path = pathlib.Path(item[0]).parts[1:]

		try :
			# os.chdir(path)
			ftp.chdir(path)
		except :
			pass

		for file in item[2]:
			if os.path.exists(UPLOAD_FOLDER):
				for f in os.listdir(UPLOAD_FOLDER):
					os.remove(os.path.join(UPLOAD_FOLDER, f))
			else :
				os.makedirs(UPLOAD_FOLDER)

			if os.path.exists(PARENT_FOLDER):	
				for f in os.listdir(PARENT_FOLDER):
					os.remove(os.path.join(PARENT_FOLDER, f))
			else :
				if not os.path.exists(PARENT_FOLDER):
					os.makedirs(PARENT_FOLDER)

			path = ftp.path.join(item[0],file)
			print(path)
			if path.endswith(ALLOWED_EXTENSIONS) :
				print(r"Copying File {0} \ {1}".format(item[0], file))
				# dest = "static/uploads"
				ftp.download(path, os.path.join(UPLOAD_FOLDER,file))
				filelist = []
				r = file.replace(" ", "")
				if(r != file):
					os.rename(UPLOAD_FOLDER + file, UPLOAD_FOLDER + r)
					file = r
				filelist.append(os.path.join(UPLOAD_FOLDER,file))
				try :
					psd, extracted_png = predict(net, Shell_File, filelist, YOLO_Model_path, PARENT_FOLDER, UPLOAD_FOLDER)
					print("**************\npsd list : ", psd)
					print("**************\nExtracted List : ", extracted_png)
					for op in psd :
						base = os.path.basename(op)
						path = pathlib.Path(os.path.join(item[0], base))
						psdpath = pathlib.Path(out).joinpath(*path.parts[1:])
						print(psdpath)
						if os.path.isfile(op):
							print(f"PSD Generated successfully {op}!!")
							ftp.upload(op, psdpath)	
							print("OP : ", op)
							print("PSDPATH : ", psdpath)
							# for file in filelist:
							if not ftp.path.isfile(psdpath) :
								errorlist.append(psdpath)
								print(f"Transfer Failed : {op}!! ")
							else :
								print(f"Transfer Successful : {psdpath}!!")
						else :
							errorlist.append(psdpath)
							print(f"PSD Generation failed for {op}!!") 
						
				except :
					print("Exception occured in Prediction !!")
					continue
			else :
				print(f"\nSkipping {path} : Extension not Allowed !\n")
				continue
	return errorlist
					
@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
	if request.method == 'POST':	
		# if folderName in ftp.nlst():
		try:
			errorlist = []
			st = time.time()
			input_path = 'input_img'
			output_path = "output_img"
			# ftp = connect_ftp()
			try :
				ftp_handle = ftputil.FTPHost('production.nextgenphotosolutions.com', 'rupesh@production.nextgenphotosolutions.com', '6Q}u}uhMDCYD')
			except :
				data = {'code' : 530,
				'message' : 'Authentication failed'
				}
				response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
				return jsonify(response.json())
		
			try :
				errorlist = download(ftp_handle, input_path, output_path, errorlist)
				et = time.time()
				elapsed_time = et - st
				ft = elapsed_time
				print('Execution time:', ft, 'secs')
				print("Errorlist : ", errorlist)
						
				data = {
					'code' : 200,
					"message" : "Thank You! We have received your confirmation.",
					"missed_files" : f"{errorlist}"
				}
				response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
				return jsonify(response.json())
			except :
				et = time.time()
				elapsed_time = et - st
				ft = elapsed_time
				print('Execution time:', ft, 'secs')
				data = {
					'code' : 400,
					'message': 'Exception Occured while processing !!',
					'missed_files': f"{errorlist}"
				}
				response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
				return jsonify(response.json())
		except:
			data = {
				'code' : 550,
				'message' : 'Files Not found'
			}
			response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
			return jsonify(response.json())

	else :
		data = {
			'code' : 400,
			'message' : 'Method Not Allowed'
		}
		response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
		return jsonify(response.json())
	
	# return render_template('download.html', filename=filename,  png_file=zipfp, psd_file=zipf, message="Extraction Successfull !!")
if __name__ == '__main__':
	app.run(host='0.0.0.0',port=8000, debug = True, threaded=True)
	
