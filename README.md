<!--
 * @Author: Sauron Wu
 * @GitHub: wutianze
 * @Email: 1369130123qq@gmail.com
 * @Date: 2019-08-27 10:48:42
 * @LastEditors: Sauron Wu
 * @LastEditTime: 2019-12-03 09:28:32
 * @Description: 
 -->
<table>
<tr>
  <td colspan="4" align="center"><img src="./images/xilinx-logo.png" width="30%"/><h1>Edge AI Tutorials</h1>
  </td>
</tr>
<tr>
<td colspan="4" align="center"><h1>Zynq 7000 DPU TRD</h1>
</td>
</tr>
</table>  

# dnndk3.0-pynqz2
- ## In this tutorial you will learn:
  1. How to use caffe model resnet50 to classify pictures using pynq-z2.
  2. How to use tensorflow model mnist to recognize hand-writing number using pynq-z2.
  3. How to train and use yolov3 in pynq-z2.
  4. How to use DNNDK-v3.0 to optimize the trained models.
  5. How to use dpu in pynq-z2 to accelerate inference.
- ## First download all the files to your pc.
  You can also download the system image of pynq-z2 we provided [in Baidu Cloud](https://pan.baidu.com/s/1gOJaoJJ8z2jf-BaLklID3Q) or [Google Cloud](https://drive.google.com/file/d/1vuwzj5Xu05-Jubmzv7Tkx-XbpLa_D_4q/view?usp=sharing)(If some files are missing, pls find them [here](https://github.com/sumilao/Zynq-7000-DPU-TRD)), it embeds DPU IP into pynq system and fixes some problems of official image. For more details, please refer to [HydraMini](https://github.com/wutianze/HydraMini).
- ## The most important files are organized as followed:
  > mnist_tf
  >> mnist_host  
  >> mnist_pynqz2  
  >> mnist-handwriting-guide.md  
  >>
  > resnet50_caffe  
  >> resnet50_host  
  >> resnet50_pynqz2  
  >> resnet50_pynqz2_guide.md
  >>
  > yolo_keras  
  >> keras-yolo3  
  >> yolo_pynqz2  
  >> take_training_imgs  
  >> yolo_pynqz2_guide.md

  The mnist_tf contains the mnist model trained by tensorflow and you can read the mnist-handwriting-guide.md to learn. The resnet50_caffe contains the resnet50 model trained by caffe and you can read the resnet50_pynqz2_guide.md to learn.  The yolo_keras provide a yolo implementation using keras, you can download the pre-trained weights of yolo from darknet.
- ## Preparation
  Before you start, you should read build-host-dnndk.md & build-pynqz2-system.md first to set your environment and do some preparation. I recommend you learn mnist_tf before running into yolo_keras.

# If you have any problem, please open an issue. If you like this project please star to support. 
## Here is a wonderful [practice](https://github.com/andre1araujo/YOLO-on-PYNQ-Z2) of this project presented by André Dias Araújo, it implements a YOLOv3 object detector on a PYNQ-Z2 PLD, making use of the Zynq-7020 to accelerate the inference and provide accurate results.
## You wanna something more exciting? See [HydraMini](https://github.com/wutianze/HydraMini)
