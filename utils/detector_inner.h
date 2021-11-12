#ifndef DETECTOR_INNER_H_
#define DETECTOR_INNER_H_

#include "detector.h"
#include "utils.h"

class DetectorInner : public Detector {
public:
    DetectorInner();

    ~DetectorInner();

    bool Init();

    bool GetDetectorResult(const cv::Mat &image, std::vector<box_prob> &boxes, char *labelpath);


};

#endif
