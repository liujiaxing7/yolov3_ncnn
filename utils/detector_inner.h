#ifndef DETECTOR_INNER_H_
#define DETECTOR_INNER_H_

#include "detector.h"

class DetectorInner : public Detector
{
public:
    DetectorInner();

    ~DetectorInner();

    bool Init();

    bool GetDetectorResult(const cv::Mat &image, std::vector<Box> &boxes);


private:
    void Display(const std::string name, const std::vector<Box> &boxInfos
                 , const cv::Mat &image) const;

public:
};


#endif
