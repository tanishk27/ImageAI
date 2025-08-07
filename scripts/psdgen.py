import subprocess
import os
import os
import signal
from subprocess import Popen, PIPE, TimeoutExpired
from time import monotonic as timer

def PSD_Generation(Shell_File, Raw_Image, Extracted_PNG, PARENT_FOLDER):	
    #filename = "output.psd"
    filename_base = os.path.basename(Raw_Image)
    filename_, file_extension = os.path.splitext(filename_base)
    file = filename_+'.psd'
    Output_PSD = os.path.join(PARENT_FOLDER, file)
    #cmd = "gimp -i -c -b (layers-to-psd (list {Raw_Image} {Extracted_PNG}) {Output_PSD) -b (gimp-quit 0)"
    #result = subprocess.call(cmd, shell=False)
    # result = subprocess.call([Shell_File, '-a', Raw_Image, '-b', Extracted_PNG, '-c', Output_PSD], stderr=subprocess.STDOUT)
    # try :
    #     proc = subprocess.Popen([Shell_File, '-a', Raw_Image, '-b', Extracted_PNG, '-c', Output_PSD], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     proc.wait()
    #     (stdout, stderr) = proc.communicate()
    #     print(stderr, stdout)
    #     print(type(proc))
    #     print("result : ", proc.returncode)
    #     return Output_PSD
    # except :
    #     if proc.returncode != 1:
    #         print(stderr)
    #     else:
    #         print("success")
    #     return Output_PSD
    start = timer()
    with Popen([Shell_File, '-a', Raw_Image, '-b', Extracted_PNG, '-c', Output_PSD], stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid) as process:
        try:
            output = process.communicate(timeout=10)[0]
        except TimeoutExpired:
            os.killpg(process.pid, signal.SIGINT) # send signal to the process group
            output = process.communicate()[0]
        finally :
            print('Elapsed seconds: {:.2f}'.format(timer() - start))
            return Output_PSD
    

#PSD_Generation(Shell_File, Raw_Image, Extracted_PNG, PARENT_FOLDER)
