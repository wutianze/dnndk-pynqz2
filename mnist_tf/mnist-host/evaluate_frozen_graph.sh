#!/bin/bash

#conda activate decent_q3

echo "#####################################"
echo "EVALUATE FROZEN GRAPH"
echo "#####################################"


python eval_graph.py \
  --graph ./freeze/frozen_graph.pb \
  --input_node images_in \
  --output_node dense_1/BiasAdd \
  --gpu 0

echo "#####################################"
echo "EVALUATE FROZEN GRAPH COMPLETED"
echo "#####################################"

