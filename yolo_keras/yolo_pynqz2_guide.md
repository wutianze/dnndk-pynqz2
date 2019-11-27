<!--
 * @Author: Sauron Wu
 * @GitHub: wutianze
 * @Email: 1369130123qq@gmail.com
 * @Date: 2019-11-06 17:34:36
 * @LastEditors: Sauron Wu
 * @LastEditTime: 2019-11-27 15:21:32
 * @Description: 
 -->
# yolo_keras_dnndk
Train your own yolo model and accelerate it using Xilinx DNNDK

# What you need:
- [keras-yolo3](https://github.com/qqwweee/keras-yolo3.git)
- [keras-yolo3-detail](https://github.com/SpikeKing/keras-yolo3-detection)
- [labelImg](https://github.com/tzutalin/labelImg.git)
- [DNNDK]()

# Generate train data:
You can see detail information in labelImg project. Here is the steps:
1. Prepare the environment, I follow the Pynthon3 + Qt5 commands in labelImg.
2. Edit `data/predefined_classes.txt` to define your own classes. For example, in FPT competition, we have:
```
person
arrowOb
coneOb
yellowOb
sidewalk
light

```
3. Open the image folder and label the images one by one. Please `Change Save Dir` to ex. `images/labels/` that labels is created in your images file. And change the style to YOLO.
4. Run `python generate_txt.py` to create `train.txt val.txt new.data` for future use.
5. Run `kmeans.py` to find anchor sizes, it will create `yolo_anchors.txt` which is specified in --output. The usage is as follows:
```
description='--cluster_number:how many clusters to make, --width --height:the image size, --filename:the merged yolo format data, --output:anchors file name'
``` 

# Train the model:
1. Set the parameters of training process in copied xx.cfg, three yolo layers and three convolutional layers above need to be changed:
```
[convolutional]
size=1
stride=1
pad=1
filters=30         #filters = 3*(classes + 5)
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=5          # classes number
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1           #memory is small then =0
```
2. Get pre-trained weight `wget https://pjreddie.com/media/files/darknet53.conv.74`
3. Run the following command to train your model.
```
./darknet detector train path-to-generated/new.data path-to-generated/yolov3.cfg darknet53.conv.74
```
4. The final weight is in the backup dir.
