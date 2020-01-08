### 
# @Author: Sauron Wu
 # @GitHub: wutianze
 # @Email: 1369130123qq@gmail.com
 # @Date: 2019-09-19 12:44:02
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2019-12-20 10:22:51
 # @Description: 
 ###
#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment
#conda activate decent_q3

# generate calibraion images and list file
#python generate_images.py

# remove existing files
rm -rf ./quantize_results


# run quantization
echo "#####################################"
echo "QUANTIZE"
echo "#####################################"
decent_q quantize \
  --input_frozen_graph ./model.pb \
  --input_nodes conv2d_1_input \
  --input_shapes ?,80,160,3 \
  --output_nodes dense_3/Relu \
  --method 1 \
  --input_fn graph_input_fn.calib_input \
  --gpu 0 \
  --calib_iter 100

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"

