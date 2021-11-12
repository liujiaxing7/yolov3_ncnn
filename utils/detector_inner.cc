#include "detector_inner.h"
#include <opencv2/highgui.hpp>
//#include "utils/timer.h"

#include "net.h"
#include "iostream"
#include "utils.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#endif

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "time.h"

ncnn::Net yolov3;

DetectorInner::DetectorInner() {
    yolov3.opt.use_vulkan_compute = true;
//    yolov3.register_custom_layer("DarknetActivation", Noop_layer_creator);
//    net.register_custom_layer("DarknetActivation", ncnn::DarknetActivation_layer_creator);
//    yolov3.register_custom_layer("Yolov3Detection", Noop_layer_creator);

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
    // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    yolov3.load_param("../mobilenetv2_yolov3.param");
    yolov3.load_model("../mobilenetv2_yolov3.bin");
}

DetectorInner::~DetectorInner() {
}

bool DetectorInner::Init() {
    return true;
}

bool DetectorInner::GetDetectorResult(const cv::Mat &image, std::vector<box_prob> &boxes, char *labelpath) {
//    std::cout << "detect run." << std::endl;
//    clock_t load_start, load_end;
//
//    load_start = clock();

//    load_end = clock();

//    printf("load_model %f seconds\n", difftime(load_end, load_start) / CLOCKS_PER_SEC);


    clock_t pretreat_start, pretreat_end;
    pretreat_start = clock();
    const int target_size = 416;

    int img_w = image.cols;
    int img_h = image.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, target_size,
                                                 target_size);

    const float mean_vals[3] = {0.0f, .0f, .0f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    pretreat_end = clock();
//    printf("pre %f seconds\n", difftime(pretreat_end, pretreat_start) / CLOCKS_PER_SEC);


    clock_t forward_start, forward_end;
    forward_start = clock();
    ncnn::Extractor ex = yolov3.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);
    forward_end = clock();
    printf("Total Detection Time: %f Seconds\n", difftime(forward_end, forward_start) / CLOCKS_PER_SEC);


    //     printf("%d %d %d\n", out.w, out.h, out.c);

    clock_t back_start, back_end;
    back_start = clock();
    boxes.clear();
    for (int i = 0; i < out.h; i++) {
        const float *values = out.row(i);

        box_prob box;
        box.class_id = values[0] - 1;
        box.image_index = *labelpath;
        box.p = values[1];
//      box.b.x = values[2] * img_w;
//      box.b.y = values[3] * img_h;
//      box.b.w = values[4] * img_w - box.b.x;
//      box.b.h = values[5] * img_h - box.b.y;

        box.b.w = values[4] - values[2];
        box.b.h = values[5] - values[3];
        box.b.x = (values[2] + box.b.w + values[2]) / 2.0;
        box.b.y = (values[3] + box.b.h + values[3]) / 2.0;

        boxes.push_back(box);
    }
    back_end = clock();
//    printf("back(maybe no back) %f seconds\n", difftime(back_end, back_start) / CLOCKS_PER_SEC);

    return 0;

    return true;
}

void Display(const std::string name, const std::vector<Box> &boxes, const cv::Mat &image) {

}