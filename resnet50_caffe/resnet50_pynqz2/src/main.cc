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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
using namespace std::chrono;

int threadnum;

mutex mutexshow;
#define KERNEL_CONV "resnet50_0"
#define CONV_INPUT_NODE "conv1"
#define CONV_OUTPUT_NODE "fc1000"

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
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, queue<string> &images) {
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
                (ext == "PNG") || (ext == "png")) {
                images.push(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kind file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, vector<string> &kinds) {
    kinds.clear();
    fstream fkinds(path);
    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }
    string kind;
    while (getline(fkinds, kind)) {
        kinds.push_back(kind);
    }

    fkinds.close();
}


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
 * @brief run classification task for Resnet_50
 *
 * @param taskConv - pointer to DPU task of Resnet_50
 * @param img - the single image to be classified
 * @param imname - the actual label of the img(for comparison)
 *
 * @return none
 */
vector<string> kinds; // Storing the label of Resnet_50
queue<string> images; // Storing the list of images
void run_resnet_50(DPUTask *taskConv, const Mat &img, string imname) {
    assert(taskConv);
    // Get the number of category in Resnet_50
    int channel = dpuGetOutputTensorChannel(taskConv, CONV_OUTPUT_NODE);
    // Get the scale of classification result
    float scale = dpuGetOutputTensorScale(taskConv, CONV_OUTPUT_NODE);
    vector<float> smRes (channel);
    int8_t* fcRes;
    // Set the input image to the DPU task
    _T(dpuSetInputImage2(taskConv, CONV_INPUT_NODE, img));
    // Processing the classification in DPU
    _T(dpuRunTask(taskConv));
    // Get the output tensor from DPU in DPU INT8 format
    DPUTensor* dpuOutTensorInt8 = dpuGetOutputTensorInHWCInt8(taskConv, CONV_OUTPUT_NODE);
    // Get the data pointer from the output tensor
    fcRes = dpuGetTensorAddress(dpuOutTensorInt8);
    // Processing softmax in DPU with batchsize=1
    _T(dpuRunSoftmax(fcRes, smRes.data(), channel, 1, scale));
    mutexshow.lock();
    // Show the top 5 classification results with their label and probability
    _T(TopK(smRes.data(), channel, 5, kinds, imname));
    mutexshow.unlock();
}

/**
 * @The entry of the whole classification process
 *
 * @param kernelconv - the pointer to DPU task of Resnet_50
 *
 * @return none
 **/
void classifyEntry(DPUKernel *kernelconv) {
    ListImages(baseImagePath, images); // Load the list of images to be classified
    if (images.size() == 0) {
        cerr << "\nError: Not images exist in " << baseImagePath << endl;
        return;
    } else {
        cout << "total image : " << images.size() << endl;
    }

    thread workers[threadnum];
    auto _start = system_clock::now();
    int size = images.size();
    for (auto i = 0; i < threadnum; i++)
    {
        workers[i] = thread([&,i]() {
            // Create DPU Tasks for Resnet_50 from DPU Kernel
            DPUTask *taskconv = dpuCreateTask(kernelconv, 0);
            while(true){
	        string imageName = images.front();
	    	if(imageName == "")
		    break;
		images.pop();

	        Mat image = imread(baseImagePath + imageName);
		// Classifying single image
                run_resnet_50(taskconv, image, imageName);
	    }
            // Destroy DPU Tasks & free resources
            dpuDestroyTask(taskconv);
        });
    }

    // Processing multi-thread classification
    for (auto &w : workers) {
        if (w.joinable()) w.join();
    }

    auto _end = system_clock::now();
    auto duration = (duration_cast<microseconds>(_end - _start)).count();
    cout << "[Time]" << duration << "us" << endl;
    cout << "[FPS]" << size*1000000.0/duration << endl;
}

void readTxt(string file)
{
    ifstream infile;
    infile.open(file.data());
    assert(infile.is_open());

    string s;
    while(getline(infile,s)){
        kinds.emplace_back(s);
    }
    infile.close();
}

/**
 * @brief Entry for running Resnet_50 neural network
 *
 */
int main(int argc ,char** argv) {

    threadnum = 5; //The number of thread with the highest efficiency
    readTxt("./word_list.txt"); //Importing the Res50 label text
    /* The main procress of using DPU kernel begin. */
    DPUKernel *kernelConv;

    TRDWarning();

    dpuOpen();
    // Create the kernel for Resnet_50
    kernelConv = dpuLoadKernel(KERNEL_CONV);
    
    // The main classification function
    //classifyEntry(kernelConv);
   
    DPUTask *taskconv = dpuCreateTask(kernelConv, 0);
    cvNamedWindow("video");
    VideoCapture cap;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 160);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
    cap.open(0);
    Mat frame;
    while(1) {
        std::cout<<"waitKey\n"; 
        char c = cvWaitKey(500);
        cap >> frame;
        imshow("video", frame);
        if(c==27) break;
        if(c==115) {
            run_resnet_50(taskconv, frame, "tmp");
        }
    }
    frame.release();
 
//    cvReleaseCapture(&cap);
    cvDestroyWindow("video");            
    // Destroy DPU Tasks & free resources
    dpuDestroyTask(taskconv);

    // Destroy the kernel of Resnet_50 after classification
    dpuDestroyKernel(kernelConv);

    dpuClose();

    TRDWarning();
    /* The main procress of using DPU kernel end. */
    return 0;
}
