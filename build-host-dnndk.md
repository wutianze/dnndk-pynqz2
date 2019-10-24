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

# This Tutorial will help you build your dnndk kit in host  
- ## Your linux version should be 14.04 or 16.04, and you should have python2.7 with pip installed if you want to use caffe.
- ## Download [dnndk](https://www.xilinx.com/member/forms/download/dnndk-eula-xef.html?filename=xlnx_dnndk_v3.0_190624.tar.gz) and extract. This guide will use dnndk_v3.0. Download our new [install.sh](./install.sh), replace the install.sh in xilinx_dnndk_v3.0/host_x86/ with ours. Then you can run `./install PynqZ2`
- ## Installing the GPU Platform Software
  The current DNNDK release can be used on the X86 host machine with or without GPU. With GPU support, DECENT is able to run faster.  
  If GPU is available in the X86 host machine, install the necessary GPU platform software in accordance with your GPU product documentation. Ensure all versions are compatible with the version of DNNDK.  
  For version information, please refer to [this](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_5/ug1327-dnndk-user-guide.pdf)
- ## Using Caffe
  1.  Installing Dependent Libraries:
  ```shell
  apt-get install -y --force-yes build-essential autoconf libtool libopenblasdev libgflags-dev libgoogle-glog-dev libopencv-dev protobuf-compiler libleveldbdev liblmdb-dev libhdf5-dev libsnappy-dev libboost-all-dev libssl-dev
  ```
  2. Install Caffe, please refer to [Caffe official website](https://caffe.berkeleyvision.org/install_apt.html)
  3. Change the $HOME/.bashrc:  
  Add two lines:
  ```sh
  export PYTHONPATH=/home/(your name)/caffe/python:$PYTHONPATH  
  export LD_LIBRARY_PATH=/home/(your name))/caffe/.build_release/lib:$LD_LIBRARY_PATH  
  ```
  Then source .bashrc
- ## Using tensorflow
  1. Install Anaconda
  Refer to [this](https://www.anaconda.com) to install conda.
  2. Install tensorflow
  ```shell
  conda create -n decent pip python=3.6
  source activate decent
  (decent)$ cd $YourDir/xilinx_dnndk_v3.0/host_x86/decent-tf/ubuntu$YourLinuxVersion/
  (decent)$ pip install ./tensorflow* # Select the right installation package for your environment
  (decent)$ pip install numpy opencv-python sklearn scipy progressbar2
  ```
