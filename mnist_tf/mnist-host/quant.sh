#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment
#conda activate decent_q3

# generate calibraion images and list file
python generate_images.py

# remove existing files
rm -rf ./quantize_results


# run quantization
echo "#####################################"
echo "QUANTIZE"
echo "#####################################"
decent_q quantize \
  --input_frozen_graph ./freeze/frozen_graph.pb \
  --input_nodes images_in \
  --input_shapes ?,28,28,1 \
  --output_nodes dense_1/BiasAdd \
  --method 1 \
  --input_fn graph_input_fn.calib_input \
  --gpu 0 \
  --calib_iter 200

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"

