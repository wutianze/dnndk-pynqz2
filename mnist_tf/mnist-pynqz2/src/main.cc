/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <dnndk/dnndk.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

int threadnum;

mutex mutexshow;
#define KERNEL_CONV "mnist"
#define CONV_INPUT_NODE "conv2d_Conv2D"
#define CONV_OUTPUT_NODE "dense_1_MatMul"

const string baseImagePath = "./image/";

#define TRDWarning()                            \
{                                    \
	cout << endl;                                 \
	cout << "####################################################" << endl; \
	cout << "Warning:                                            " << endl; \
	cout << "The DPU in this TRD can only work 8 hours each time!" << endl; \
	cout << "Please consult Sales for more details about this!   " << endl; \
	cout << "####################################################" << endl; \
	cout << endl;                                 \
}

//#define SHOWTIME
#ifdef SHOWTIME
#define _T(func)                                                              \
{                                                                             \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeTest]" << left << setw(30) << tmp;                      \
        cout << left << setw(10) << duration << "us" << endl;                 \
}
#else
#define _T(func) func;
#endif

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, vector<string> &vkinds, string name) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    cout << "\nLoad image: " << name << endl;

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        printf("[Top %d] prob = %-8f  name = %s\n", i, d[ki.second], vkinds[ki.second].c_str());
        q.pop();
    }
    return;
}

/**
 * @brief Entry for running Resnet_50 neural network
 *
 */
int main(int argc ,char** argv) {

    /* The main procress of using DPU kernel begin. */
    DPUKernel *kernelConv;

    TRDWarning();

    dpuOpen();
    // Create the kernel for mnist
    kernelConv = dpuLoadKernel(KERNEL_CONV);

    DPUTask *taskMnist = dpuCreateTask(kernelConv,0);
    
    /*
    cout << dpuGetInputTensorScale(taskMnist, CONV_INPUT_NODE) << endl;
    cout << dpuGetInputTensorHeight(taskMnist, CONV_INPUT_NODE) << endl;
    cout << dpuGetInputTensorWidth(taskMnist, CONV_INPUT_NODE) << endl;
    cout << dpuGetInputTensorChannel(taskMnist, CONV_INPUT_NODE) << endl;
    */

    Mat image = imread("../2.bmp",0);
    float meanV[1] = {0.5f};
    _T(dpuSetInputImage(taskMnist, CONV_INPUT_NODE, image,meanV))
    _T(dpuRunTask(taskMnist));
    float scale = dpuGetOutputTensorScale(taskMnist, CONV_OUTPUT_NODE);
    cout << dpuGetOutputTensorHeight(taskMnist, CONV_OUTPUT_NODE) << endl;
    cout << dpuGetOutputTensorWidth(taskMnist, CONV_OUTPUT_NODE) << endl;
    int channel = dpuGetOutputTensorChannel(taskMnist, CONV_OUTPUT_NODE);
    vector<float> smRes(channel);
    int8_t* fcRes;
    DPUTensor* dpuOutTensorInt8 = dpuGetOutputTensorInHWCInt8(taskMnist, CONV_OUTPUT_NODE);
    fcRes = dpuGetTensorAddress(dpuOutTensorInt8);
    _T(dpuRunSoftmax(fcRes, smRes.data(),channel,1,scale));
    vector<string>kinds = {"0","1","2","3","4","5","6","7","8","9"};

    _T(TopK(smRes.data(),channel,3,kinds,"2.bmp"));
    // The main classification function
    // classifyEntry(kernelConv);
    // Destroy the kernel of Resnet_50 after classification
    dpuDestroyKernel(kernelConv);

    dpuClose();

    TRDWarning();
    /* The main procress of using DPU kernel end. */
    return 0;
}
