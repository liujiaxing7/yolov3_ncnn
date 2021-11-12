#include "detector.h"
#include "detector_inner.h"

Detector *Create() {
    return static_cast<Detector *>(new DetectorInner());
}

void Destroy(Detector *detectorPtr) {
    if (detectorPtr != nullptr) {
        delete detectorPtr;
        detectorPtr = nullptr;
    }
}
