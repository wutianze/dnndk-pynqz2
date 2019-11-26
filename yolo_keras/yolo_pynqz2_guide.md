<!--
 * @Author: Sauron Wu
 * @GitHub: wutianze
 * @Email: 1369130123qq@gmail.com
 * @Date: 2019-11-06 17:34:36
 * @LastEditors: Sauron Wu
 * @LastEditTime: 2019-11-26 10:57:29
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
3. Open the image folder and label the images one by one.
4. Run `kmeans.py` to find anchor sizes. The usage is as follows:
```
description='--cluster_num:how many clusters to make, --width --height:the image size, --filename:the merged yolo format data, --output:anchors file name'
``` 

# Train the model:
1. Set the parameters of training process:
```
```
