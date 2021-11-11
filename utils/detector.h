//////////////////////////////////////////////////////////////////////
///  @file     detector.h
///  @brief    object detect based on deep learning for i18R
///            small wash robot
///  Details.
///
///  @author   sunhao
///  @version  1.7.0
///  @date     2021.04.19
///
///  revision statement:
///            1.0.0 beta version
///            1.1.0 add pose interface; rename interface
///            2021.10.27 1.7.0 add new error message interface
//////////////////////////////////////////////////////////////////////

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <functional>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.h"

struct Box
{
//    cv::Rect_<float> rect;
    float x, y, w, h;
};
typedef struct {
    Box b;
    float p;
    int class_id;
    char image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

class Detector
{
public:
    Detector()
    {}

    virtual ~Detector()
    {}

    struct detection;
    typedef struct detection detection;
    /**
     * @brief init the deep learning model
     * @param[in] detectParam : module param about config and so on
     *                          @see I18RPublicBaseTypes/detector/detect_types.h
     *
     * @return init status. [true] success, [false] fail
     */
    virtual bool Init() = 0;

    virtual bool GetDetectorResult(const cv::Mat &image, std::vector<box_prob> &boxes, char *labelpath) = 0;

    virtual float validate_detector_map(std::vector<std::vector<box_prob>> &boxes,std::vector<std::vector<box_label>> &truth1, float thresh_calc_avg_iou, const float iou_thresh, int map_points)= 0;


};

/**
 * @brief create the module pointer
 * @param moduleParam : the param about the robot e.g. camera param
 * @return the moudle pointer
 * should be use @see `Destroy` to release when useless
 */
Detector *Create();

void Destroy(Detector *detectorPtr);

#endif
