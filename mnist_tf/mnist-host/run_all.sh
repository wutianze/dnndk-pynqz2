#!/bin/bash


# clean up previous log files
rm -rf *.log


source ./train.sh | tee train.log

source ./freeze.sh | tee freeze.log

source ./evaluate_frozen_graph.sh | tee evaluate_frozen_graph.log

source ./quant.sh | tee quant.log

source ./evaluate_quantized_graph.sh | tee evaluate_quantized_graph.log

source ./compile.sh | tee compile.log


