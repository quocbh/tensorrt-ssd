#include <iostream>
#include <algorithm>
#include "tensorRTplugin/tensorNet.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "util/cuda/cudaRGB.h"
#include "util/loadImage.h"
#include <chrono>

using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace cv;

const char* model  = "/home/nvidia/Documents/SSD_300x300/ssd_deploy_iplugin.prototxt";
const char* weight = "/home/nvidia/Documents/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel";
const char* label  = "/home/nvidia/Documents/SSD_300x300/labelmap_coco.prototxt";

static const uint32_t BATCH_SIZE = 1;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT = "detection_out";

class Timer {
 public:
  void tic() {
    start_ticking_ = true;
    start_ = std::chrono::high_resolution_clock::now();
  }
  double toc() {
    if (!start_ticking_)return -1;
    end_ = std::chrono::high_resolution_clock::now();
    start_ticking_ = false;
    double t = std::chrono::duration<double, std::milli>(end_ - start_).count();
    std::cout << "Time: " << t << " ms" << std::endl;
    return t;
  }
 private:
  bool start_ticking_ = false;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

void CheckImageSize(cv::Mat* image, std::size_t size) {
  if (image->rows != size || image->cols != size)
    cv::resize(*image, *image, cv::Size(size, size));
}

cudaError_t cudaPreImageNetMean( float3* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value);

float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}

vector<std::string> loadLabelInfo(const char* filename)
{   
    assert(filename);
    std::vector<std::string> labelInfo;

    FILE* f = fopen(filename, "r");
    if( !f )
    {   
        printf("failed to open %s\n", filename);
        assert(0);
    }
    
    char str[512];
    char name[512];
    while( fgets(str, 512, f) != NULL )
    {     
        if(str[2] == 'l')
        {   
            fgets(name, 512, f);

            const int len = strlen(name);
            std::string label;
            for (int i=17; i<len-2; i++) {
                label+=name[i];
            }
            labelInfo.push_back(label);
        }
        else continue;
    }
    fclose(f);
    return labelInfo;
}


int main()
{

    cout << "Hello, World!" << std::endl;
    //VideoCapture cap("/home/nvidia/Downloads/male.mp4");
    const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! \
			nvvidconv flip-method=0 ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";

    cv::VideoCapture cap(gst);
    if(!cap.isOpened())
    {
        cout<<"There is no video in this location"<<endl;
        return -1;
    }
    namedWindow("VideoCapture", WINDOW_AUTOSIZE);

    vector<std::string> labelInfo = loadLabelInfo(label);

    TensorNet tensorNet;
    tensorNet.caffeToTRTModel( model, weight, std::vector<std::string>{ OUTPUT}, BATCH_SIZE);
    tensorNet.createInference();

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut    = tensorNet.getTensorDims(OUTPUT);
    ///cout << "INPUT Tensor Shape is: C: "  <<dimsData.c()<< "  H: "<<dimsData.h()<<"  W:  "<<dimsData.w()<<endl;

    float* data    = allocateMemory( dimsData, (char*)"input blob");
    float* output  = allocateMemory( dimsOut, (char*)"output blob");

    int height = 300;
    int width  = 300;

    Mat frame;
    Mat frame_float;
    //frame = cv::imread("/home/nvidia/Downloads/cat.jpg", IMREAD_COLOR);
    void* imgCPU;
    void* imgCUDA;
    Timer timer;
    double time;
    while (true) 
    {
        timer.tic();
        cap.read(frame);
        resize(frame, frame, Size(300,300));
        const size_t size = width * height * sizeof(float3);

        if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
        {
            cout <<"Cuda Memory allocation error occured."<<endl;
            return false;
        }
        if( !loadImageBGR( frame , (float3**)&imgCPU, (float3**)&imgCUDA, &height, &width))
        {
            printf("failed to load image '%s'\n", "Image");
            return 0;
        }
        void* buffers[] = {imgCUDA, output};
        tensorNet.imageInference( buffers, 2, BATCH_SIZE);
        time=timer.toc();

	for (int k=0; k<200; k++)
        {               
	    if (output[7*k+1] ==-1) 
            {
                 break;
	    }

            float xmin = 300 * output[7*k + 3];
            float ymin = 300 * output[7*k + 4];
            float xmax = 300 * output[7*k + 5];
            float ymax = 300 * output[7*k + 6];             
            cv::Point2f a = Point2f(xmin, ymin);
            cv::Point2f b = Point2f(xmax, ymax);

            string conf = to_string(output[7*k + 2]); 
            rectangle(frame, a, b, Scalar(0.0, 255.0, 255.0));
            cv::putText(frame,labelInfo[output[7*k + 1]] + ":" + conf, Point2f(xmin, ymin+ 15),FONT_HERSHEY_DUPLEX, 0.5, (0, 253, 255), 1, 2);
            //cv::putText(frame,to_string(1000/time), Point2f(0, 15),FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, 2);
        }
        imshow("VideoCapture", frame);
        waitKey(30);
        CUDA(cudaFreeHost(imgCPU));
    }
    CUDA(cudaFreeHost(imgCPU));
    tensorNet.destroy();
    return 0;
}
