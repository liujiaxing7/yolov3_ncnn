#include <iostream>

#include "utils/detector.h"
#include "utils/detector_inner.h"
#include "utils/utils.h"
#include <unistd.h>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    std::string inputDir = argv[1];
    Detector *detector = Create();


    std::string imagesTxt = inputDir + "/image.txt";
    std::vector<std::string> imageNameList;
    std::vector<std::string> lidarNameList;

    ReadFile(imagesTxt, imageNameList);
    const size_t size = imageNameList.size();
    printf("size:\n",size);

    std::vector<Box> objects1;
    for (size_t i = 0; i < size; ++i) {

        auto imageName = imageNameList.at(i);
        std::string imagePath(inputDir + "/" + imageName);

        cv::Mat m = cv::imread(imagePath, 1);

        if (m.empty()) {
            fprintf(stderr, "cv::imread %s failed\n", imageName.c_str());
            return -1;
        }

        std::vector<Box> objects;
        detector->GetDetectorResult(m,objects);
//        detect_yolov3(m, objects);
//        objects1.push_back(objects);


//        draw_objects(m, objects);
    }

    return 0;
}