//
// Created by fandong on 2021/11/8.
//

#ifndef YOLOV3_NCNN_YOLOV3_H
#define YOLOV3_NCNN_YOLOV3_H

#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>
#include "time.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
//class Noop : public ncnn::Layer {};
//DEFINE_LAYER_CREATOR(Noop)
static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects)
{
    clock_t load_start,load_end;

    load_start=clock();
    ncnn::Net yolov3;

    yolov3.opt.use_vulkan_compute = true;
//    yolov3.register_custom_layer("DarknetActivation", Noop_layer_creator);
//    net.register_custom_layer("DarknetActivation", ncnn::DarknetActivation_layer_creator);
//    yolov3.register_custom_layer("Yolov3Detection", Noop_layer_creator);

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
    // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    yolov3.load_param("mobilenetv2_yolov3.param");
    yolov3.load_model("mobilenetv2_yolov3.bin");
    load_end=clock();

    printf("load_model %f seconds\n", difftime(load_end,load_start)/ CLOCKS_PER_SEC );


    clock_t pretreat_start,pretreat_end;
    pretreat_start=clock();
    const int target_size = 416;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {0.0f, .0f, .0f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    pretreat_end=clock();
    printf("pre %f seconds\n", difftime(pretreat_end,pretreat_start)/ CLOCKS_PER_SEC  );



    clock_t forward_start,forward_end;
    forward_start=clock();
    ncnn::Extractor ex = yolov3.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);
    forward_end=clock();
    printf("forward %f seconds\n", difftime(forward_end,forward_start)/ CLOCKS_PER_SEC  );

    //     printf("%d %d %d\n", out.w, out.h, out.c);

    clock_t back_start,back_end;
    back_start=clock();
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }
    back_end=clock();
    printf("back(maybe no back) %f seconds\n", difftime(back_end,back_start)/ CLOCKS_PER_SEC  );

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "person", "escalator", "escalator_handrails", "person_dummy"

    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}



#endif //YOLOV3_NCNN_YOLOV3_H
