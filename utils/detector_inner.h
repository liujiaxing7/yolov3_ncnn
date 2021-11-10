#ifndef DETECTOR_INNER_H_
#define DETECTOR_INNER_H_

#include "detector.h"
#include "utils.h"

class DetectorInner : public Detector {
public:
    DetectorInner();

    ~DetectorInner();

    bool Init();

    struct detection;
    typedef struct detection detection;

    bool GetDetectorResult(const cv::Mat &image, std::vector<Box> &boxes);

    float
    validate_detector_map(std::vector<Box> &boxes, box_label *truth, float thresh_calc_avg_iou, const float iou_thresh,
                          int map_points);

    int detections_comparator(const void *pa, const void *pb);

private:
    void Display(const std::string name, const std::vector<Box> &boxInfos, const cv::Mat &image) const;

    void free_detections(detection *dets, int n);
};
#endif
