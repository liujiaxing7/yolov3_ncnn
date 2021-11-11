#ifndef DETECTOR_INNER_H_
#define DETECTOR_INNER_H_

#include "detector.h"
#include "utils.h"
struct detection;
typedef struct detection detection;
int detections_comparator(const void *pa, const void *pb);
void free_detections(detection *dets, int n);
class DetectorInner : public Detector {
public:
    DetectorInner();

    ~DetectorInner();

    bool Init();

    bool GetDetectorResult(const cv::Mat &image, std::vector<box_prob> &boxes,char *labelpath);

    float
    validate_detector_map(std::vector<std::vector<box_prob>> &boxes, box_label *truth,int *nums_labels, float thresh_calc_avg_iou, const float iou_thresh,
                          int map_points);

private:
    void Display(const std::string name, const std::vector<Box> &boxInfos, const cv::Mat &image) const;


};
#endif
