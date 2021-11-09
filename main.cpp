#include "utils/yolov3.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>


int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char *imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov3(m, objects);

    draw_objects(m, objects);

    return 0;
}
