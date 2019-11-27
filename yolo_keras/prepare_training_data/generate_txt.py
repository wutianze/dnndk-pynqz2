'''
@Author: Sauron Wu
@GitHub: wutianze
@Email: 1369130123qq@gmail.com
@Date: 2019-11-27 13:46:08
@LastEditors: Sauron Wu
@LastEditTime: 2019-11-27 15:45:13
@Description: 
'''
import glob
import os
import numpy as np
dir_now = os.getcwd()
path = dir_now + "/images"
val_split = 0.1
img_lines = glob.glob(path + "/*.jpg")
np.random.seed(10101)
np.random.shuffle(img_lines)
np.random.seed(None)
num_val = int(len(img_lines)*val_split) + 1

os.mkdir('./generated')
train_txt = open('generated/train.txt','w')
val_txt = open('generated/val.txt','w')
for i in img_lines[num_val:]:
    train_txt.write(i+'\n')
train_txt.close()
for i in img_lines[:num_val]:
    val_txt.write(i+'\n')
val_txt.close()

label_path = path + '/labels/classes.txt'
labels = open(label_path,'r')
class_num = 0
for line in labels:
    class_num = class_num + 1
print("classes num:%d"%class_num)
data_file = open('generated/new.data','w')
data_file.write("classes="+str(class_num)+'\n')
data_file.write("train="+dir_now+'/generated/train.txt\n')
data_file.write("valid="+dir_now+'/generated/val.txt\n')
data_file.write("names="+label_path+'\n')
data_file.write("backup="+dir_now+'/backup\n')
os.mkdir('./backup')
labels.close()
data_file.close()

