###
 # @Author: Sauron Wu
 # @GitHub: wutianze
 # @Email: 1369130123qq@gmail.com
 # @Date: 2019-10-15 15:36:14
 # @LastEditors: Sauron Wu
 # @LastEditTime: 2019-12-20 11:39:21
 # @Description: 
 ###
#!/bin/bash

# delete previous results
rm -rf ./compile


#conda activate decent_q3


# Compile
echo "#####################################"
echo "COMPILE WITH DNNC"
echo "#####################################"
dnnc \
       --parser=tensorflow \
       --frozen_pb=./quantize_results/deploy_model.pb \
       --dpu=1152FA \
       --cpu_arch=arm32 \
       --output_dir=compile \
       --save_kernel \
       --mode normal \
       --net_name=testModel

echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"

