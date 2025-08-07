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
import ftplib
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
import faulthandler; faulthandler.enable()
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

import faulthandler
faulthandler.enable()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def divide_chunks(l, n):
    # looping till length l
	for i in range(0, len(l), n):
		yield l[i:i + n]     

def connect_ftp():
	try :
		ftp = ftplib.FTP('production.nextgenphotosolutions.com', 'ai_work@production.nextgenphotosolutions.com', '12Z}u;!TkQtf%w4', timeout=5000)
		return ftp
	except :
		data = {'code' : 530,
				'message' : 'Authentication failed'
		}
		response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
		return jsonify(response.json())
		
@app.route('/')
def upload_form():
	#flash('Image successfully uploaded !!')
	return render_template('input.html')

@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
	if request.method == 'POST':	
		# if folderName in ftp.nlst():
		try:
			errorlist = []
			st = time.time()
			folderName = 'input_img'
			input_path = '/'+folderName+'/'
			output_path = "/output_img/"
			ftp = connect_ftp()

			if folderName in ftp.nlst():
				ftp.cwd(input_path) 
				ftp.encoding = "utf-8"
				files_ = ftp.nlst()
				# print(files_)
				files_.sort()
				
				files = [f for f in files_ if allowed_file(f)]	
				if len(files) > 0:		
					chunks = list(divide_chunks(files, 1))
					print(chunks)
					
					for chunk in chunks:
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
						filelist =[]
						for f in chunk:
							print(f)
							filesize = ftp.size(f)
							localfile = open(UPLOAD_FOLDER + f, 'wb')
							ftp.retrbinary('RETR ' + f, localfile.write)
							localfile.close()
							r = f.replace(" ", "")
							if(r != f) :
								os.rename(UPLOAD_FOLDER + f,UPLOAD_FOLDER + r)
								f = r
							if os.path.getsize(UPLOAD_FOLDER + f) == filesize:
								filelist.append(UPLOAD_FOLDER + f)
								print(f + ' has been completely received.')
							else:
								try :
									print(f + ' transfer incomplete. Trying to load again !!')
									ftp.retrbinary('RETR ' + f, localfile.write)
								except :
									print(f + ' transfer failed.')
									continue
						# ftp.quit()

						# psd, extracted_png = Final_PSD(net, Shell_File, filelist, YOLO_Model_path, PARENT_FOLDER, UPLOAD_FOLDER)
						try :
							psd, extracted_png = predict(net, Shell_File, filelist, YOLO_Model_path, PARENT_FOLDER, UPLOAD_FOLDER)
							print("**************\npsd list : ", psd)
							print("**************\nExtracted List : ", extracted_png)
							# psdn = []
							# pngn = []
							filen = []
							# for file in psd :
							# 	file = os.path.splitext(os.path.basename(file))[0]
							# 	psdn.append(file)
							# for file in extracted_png :
							# 	file = os.path.splitext(os.path.basename(file))[0]
							# 	pngn.append(file)
							# for file in filelist:
							# 	base = os.path.splitext(os.path.basename(file))[0]
							# 	path = base + '.psd'
							# 	if os.path.isfile(os.path.join(PARENT_FOLDER, path)):
							# 		errorlist.append(os.path.basename(file))
							for file in chunk :
									file = os.path.splitext(os.path.basename(file))[0]
									filen.append(file)
							# psdn.sort()
							# pngn.sort()
							# filen.sort()
							# if psdn != pngn :
							# 	text = "PSD's not generated for some files"
							# 	errorlist.append(text)
							# if filen != pngn or filen != psdn :
							# 	text = "Some files didn't extracted "
							# 	errorlist.append(text)
							for file in filelist:
								base = os.path.splitext(os.path.basename(file))[0]
								path = base + '.psd'
								print('in errorlist', path)
								if not os.path.exists(os.path.join(PARENT_FOLDER, path)):
									errorlist.append(path)
							# ftp = connect_ftp()
							ftp.cwd(".")
							ftp.cwd(output_path)
							##Upload results to server
							
							for output in psd :
								try :
									with open(output, "rb") as file:
										psdpath = os.path.basename(output)
										# Command for Uploading the file "STOR filename"
										ftp.storbinary(f"STOR {psdpath}", file)
										# globallist.append(psdpath)
										file.close()
								except :
									print("\n*********\nFile missed !\n*********\n")
									psdpath = os.path.basename(output)
									errorlist.append(psdpath)
									continue
							# ftp.quit()
						except :
							continue
												
						# print('Batch completed !!')
						# ftp.close()
						# ftp = connect_ftp()
						ftp.cwd(".")
						ftp.cwd(input_path)
						print('Batch Completed !!')
						#errorlist.append("Batch Completed !!")
					et = time.time()
					# get the execution time
					elapsed_time = et - st
					ft = elapsed_time
					# ftp.quit()
					print('Execution time:', ft, 'secs')

					print("\nErrorlist : ", errorlist)
					# if len(filen) != len(psd) :
					# 	data = {
					# 		'code' : 400,
					# 		"message" : "Thank You! We have received your confirmation. But Some files didn't extracted !!",
					# 		"missed_files" : f"{errorlist}"
					# 	}
					# 	response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
					# 	return jsonify(response.json())
					# else :
					data = {
						'code' : 200,
						"message" : "Thank You! We have received your confirmation.",
						"missed_files" : f"{errorlist}"
					}
					response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
					return jsonify(response.json())
				else :
						data = {
							'code' : 550,
							'message' : 'Files Not found'
						}
						response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
						return jsonify(response.json())
			else :
				data = {
					'code' : 412,
					'message' : 'Directory Not Found'
				}
				response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
				return jsonify(response.json())
		except ftplib.error_perm as resp:
			if str(resp) == "550 No files found":
				data = {
					'code' : 550,
					'message' : 'Files Not found'
				}
				response = requests.post(url="https://production.nextgenphotosolutions.com/aiservices/ai_placed_psd", json=data)
				return jsonify(response.json())
			else:
				raise
		
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
	
