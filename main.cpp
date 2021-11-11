#include <iostream>

#include "utils/detector.h"
#include "utils/detector_inner.h"
#include "utils/utils.h"
#include <unistd.h>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    std::string inputDir = argv[1];
    Detector *detector = Create();


    std::string imagesTxt = inputDir + "/image.txt";
    std::vector<std::string> imageNameList;
    std::vector<std::string> lidarNameList;

    ReadFile(imagesTxt, imageNameList);
    const size_t size = imageNameList.size();

    std::vector<std::vector<box_prob>> boxes;
    
    int count1=0;
    box_label* truth_all = (box_label*)xcalloc(1, sizeof(box_label));
    box_label * truth;
    int nums_labels;
    for (size_t i = 0; i < size; ++i) {

        auto imageName = imageNameList.at(i);
        std::string imagePath1( imageName);
        char *imagePath= const_cast<char *>(imagePath1.data());
//        char *labelPath=imagePath.replace(imagePath.find("JPEGImages"), 3, "labels");
        char labelpath[4096];
        replace_image_to_label(imagePath, labelpath);

        cv::Mat m = cv::imread(imagePath1, 1);

        if (m.empty()) {
            fprintf(stderr, "cv::imread %s failed\n", imageName.c_str());
            return -1;
        }

        std::vector<box_prob> box;
        detector->GetDetectorResult(m,box,labelpath);
        boxes.push_back(box);

        nums_labels=0;
        truth=read_boxes(labelpath,&nums_labels);

        for(int i=0;i< nums_labels;++i){
            truth_all= (box_label*)xrealloc(truth_all, (count1+ 1) * sizeof(box_label));

            truth_all[count1].track_id = count1 + 0;
            //printf(" boxes[count1].track_id = %d, count1 = %d \n", boxes[count1].track_id, count1);
            truth_all[count1].id = truth[i].id;
            truth_all[count1].x = truth[i].x;
            truth_all[count1].y = truth[i].y;
            truth_all[count1].h = truth[i].h;
            truth_all[count1].w = truth[i].w;
            truth_all[count1].left   = truth[i].x - truth[i].w/2;
            truth_all[count1].right  = truth[i].x + truth[i].w/2;
            truth_all[count1].top    = truth[i].y - truth[i].h/2;
            truth_all[count1].bottom = truth[i].y + truth[i].h/2;

            ++count1;
            
        }


//        detect_yolov3(m, objects);
//        objects1.push_back(objects);


//        draw_objects(m, objects);
    }
    detector->validate_detector_map(boxes, truth, &nums_labels, 0.5, 0.25, 0);
    printf(" ");

    return 0;
}