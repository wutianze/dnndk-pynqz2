#!/bin/bash
net=resnet50
model_dir=decent_output
output_dir=dnnc_output

echo "Compiling network: ${net}"

dnnc --prototxt=${model_dir}/deploy.prototxt     \
       --caffemodel=${model_dir}/deploy.caffemodel \
       --output_dir=${output_dir}                  \
       --net_name=${net}                           \
       --dpu=1152FA                                 \
       --cpu_arch=arm32                         
