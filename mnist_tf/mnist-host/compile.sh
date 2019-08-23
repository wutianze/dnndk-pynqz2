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
       --net_name=mnist

echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"

