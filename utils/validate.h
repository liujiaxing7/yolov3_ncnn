//
// Created by fandong on 2021/11/11.
//

#ifndef YOLOV3_NCNN_VALIDATE_H
#define YOLOV3_NCNN_VALIDATE_H

#include "detector.h"
#include "utils.h"
#include "detector_inner.h"

//struct Box
//{
////    cv::Rect_<float> rect;
//    float x, y, w, h;
//};
//typedef struct {
//    Box b;
//    float p;
//    int class_id;
//    char image_index;
//    int truth_flag;
//    int unique_truth_index;
//} box_prob;


struct detection;

typedef struct detection detection;

int detections_comparator(const void *pa, const void *pb);

void free_detections(detection *dets, int n);

void validate_detector_map(std::vector<std::vector<box_prob>> &boxes, std::vector<std::vector<box_label>> &truth1,
                            float thresh_calc_avg_iou, const float iou_thresh,
                            int map_points);
//class validate {
//public:
//
//
//
//};


#endif //YOLOV3_NCNN_VALIDATE_H
