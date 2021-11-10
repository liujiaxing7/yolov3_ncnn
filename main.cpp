#include <iostream>
#include "utils/yolov3.h"
#include "utils/utils.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

int main(int argc, char** argv)
//int main()
{

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
  const char* imagepath = argv[1];

    std::string inputDir = argv[1];
//    std::string inputDir = "/home/fandong/images";
    std::string imagesTxt = inputDir + "/sub.txt";
    std::vector<std::string> imageNameList;
    std::vector<std::string> lidarNameList;

    ReadFile(imagesTxt, imageNameList);
    const size_t size = imageNameList.size();
    printf("size:\n",size);

    for (size_t i = 0; i < size; ++i) {

        auto imageName = imageNameList.at(i);
        std::string imagePath(inputDir + "/" + imageName);

        cv::Mat m = cv::imread(imagePath, 1);

        if (m.empty()) {
            fprintf(stderr, "cv::imread %s failed\n", imageName.c_str());
            return -1;
        }

        std::vector<Object> objects;
        detect_yolov3(m, objects);


//        draw_objects(m, objects);
    }

    return 0;
}