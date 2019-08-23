#####################################################
# Converts MNIST numpy arrays to image files
# for calibration during quantization
#####################################################

import os
import shutil
import cv2

from keras.datasets import mnist


#####################################################
# Set up directories
#####################################################

SCRIPT_DIR = os.getcwd()
CALIB_DIR = os.path.join(SCRIPT_DIR, 'calib_dir')
IMAGE_LIST_FILE = 'calib_list.txt'


if (os.path.exists(CALIB_DIR)):
    shutil.rmtree(CALIB_DIR)
os.makedirs(CALIB_DIR)
print('Directory', CALIB_DIR, 'created') 



#####################################################
# Get the dataset using Keras
#####################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# create file for list of calibration images
f = open(os.path.join(CALIB_DIR, IMAGE_LIST_FILE), 'w')


#####################################################
# convert test dataset into image files
#####################################################

for i in range(len(x_test)):
    cv2.imwrite(os.path.join(CALIB_DIR,'x_test_'+str(i)+'.png'), x_test[i])
    f.write('x_test_'+str(i)+'.png\n')

f.close()

print ('FINISHED GENERATING CALIBRATION IMAGES')

