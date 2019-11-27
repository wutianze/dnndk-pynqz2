'''
@Author: Sauron Wu
@GitHub: wutianze
@Email: 1369130123qq@gmail.com
@Date: 2019-11-27 13:46:08
@LastEditors: Sauron Wu
@LastEditTime: 2019-11-27 13:59:11
@Description: 
'''
import glob
import os
import numpy as np
path = os.getcwd() + "/images"
val_split = 0.1
txt_lines = glob.glob(path + "/*.txt")
img_lines = glob.glob(path + "/*.jpg")
np.random.seed(10101)
np.random.shuffle(txt_lines)
np.random.seed(10101)
np.random.shuffle(img_lines)
np.random.seed(None)
num_val = int(len(txt_lines)*val_split)
num_train = len(txt_lines) - num_val

train_txt = open('train.txt','w')
val_txt = open('val.txt','w')
for i in img_lines[num_val:]:
    train_txt.write(i+'\n')
train_txt.close()
for i in img_lines[:num_val]:
    val_txt.write(i+'\n')
val_txt.close()
