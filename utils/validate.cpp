//
// Created by fandong on 2021/11/11.
//

#include "validate.h"
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

typedef struct detection {
    Box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
    float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
    int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
    float *embeddings;  // embeddings for tracking
    int embedding_size;
    float sim;
    int track_id;
} detection;

float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(Box a, Box b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union(Box a, Box b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(Box a, Box b) {
    //return box_intersection(a, b)/box_union(a, b);

    float I = box_intersection(a, b);
    float U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}

int detections_comparator(const void *pa, const void *pb) {
    box_prob a = *(const box_prob *) pa;
    box_prob b = *(const box_prob *) pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

//void free_detections(detection *dets, int n)
//{
//    int i;
//    for (i = 0; i < n; ++i) {
//        free(dets[i].prob);
//        if (dets[i].uc) free(dets[i].uc);
//        if (dets[i].mask) free(dets[i].mask);
//        if (dets[i].embeddings) free(dets[i].embeddings);
//    }
//    free(dets);
//}

void validate_detector_map(std::vector<std::vector<box_prob>> &boxes_all,
                            std::vector<std::vector<box_label>> &truth1, float thresh_calc_avg_iou,
                            const float iou_thresh, int map_points) {

    FILE *reinforcement_fd = NULL;

    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");
//    char **names = get_labels_custom(name_list, &names_size);
//    char **names = get_labels_custom(name_list, &names_size);
    static const char *names[] = {
            "person", "escalator", "escalator_handrails", "person_dummy", "5", "6"

    };
    //定义一些超参数和阈值
    int classes = 6;

    int m = boxes_all.size();
//    int m= std::count(boxes.begin(),boxes.end());
    int i0 = 0;
    int t;

    const float thresh = .25;
    const float nms = .45;
    //const float iou_thresh = 0.5;

    int nthreads = 4;
    int num_labels = classes;

//    load_args args = { 0 };
    int w = 416;
    int h = 416;
    int c = 3;
//    letter_box = net.letter_box;
//    if (letter_box) args.type = LETTERBOX_DATA;
//    else args.type = IMAGE_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    //定义预测box和标签box
    box_prob *detections = (box_prob *) xcalloc(1, sizeof(box_prob));

    int detections_count = 0;
    int unique_truth_count = 0;

    int *truth_classes_count = (int *) xcalloc(classes, sizeof(int));

    // For multi-class precision and recall computation
    float *avg_iou_per_class = (float *) xcalloc(classes, sizeof(float));
    int *tp_for_thresh_per_class = (int *) xcalloc(classes, sizeof(int));
    int *fp_for_thresh_per_class = (int *) xcalloc(classes, sizeof(int));
    char **file_paths = (char **) xcalloc(m, 100);

    for (int i1 = 0; i1 < m; i1++) {


        std::vector<box_prob> boxes = boxes_all[i1];
        int nums_prob = boxes.size();

        std::vector<box_label> truth = truth1[i1];

        int nums_labels = truth.size();
//         = truth1;

        int j;
        for (j = 0; j < nums_labels; ++j) {
            truth_classes_count[truth[j].id]++;
        }
        const int checkpoint_detections_count = detections_count;

        int i;
        for (i = 0; i < nums_prob; ++i) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                float prob = boxes[i].p;
                if (prob > 0 && boxes[i].class_id == class_id) {
                    detections_count++;
                    detections = (box_prob *) xrealloc(detections, detections_count * sizeof(box_prob));
                    detections[detections_count - 1].b = boxes[i].b;
                    detections[detections_count - 1].p = prob;
                    detections[detections_count - 1].image_index = 0;
                    detections[detections_count - 1].class_id = class_id;
                    detections[detections_count - 1].truth_flag = 0;
                    detections[detections_count - 1].unique_truth_index = -1;

                    int truth_index = -1;
                    float max_iou = 0;
                    for (j = 0; j < nums_labels; ++j) {
                        Box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
                        //printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n",
                        //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
                        float current_iou = box_iou(boxes[i].b, t);
//                        printf("iou: %f \n",current_iou);
                        if (current_iou > iou_thresh && class_id == truth[j].id) {
//                            printf("66666666");
                            if (current_iou > max_iou) {

                                max_iou = current_iou;
                                truth_index = unique_truth_count + j;
                            }
                        }
                    }

                    // best IoU
                    if (truth_index > -1) {
                        detections[detections_count - 1].truth_flag = 1;
                        detections[detections_count - 1].unique_truth_index = truth_index;
                    }

                    // calc avg IoU, true-positives, false-positives for required Threshold
                    if (prob > thresh_calc_avg_iou) {
                        int z, found = 0;
                        for (z = checkpoint_detections_count; z < detections_count - 1; ++z) {
                            if (detections[z].unique_truth_index == truth_index) {
                                found = 1;
                                break;
                            }
                        }

                        if (truth_index > -1 && found == 0) {
                            avg_iou += max_iou;
                            ++tp_for_thresh;
                            avg_iou_per_class[class_id] += max_iou;
                            tp_for_thresh_per_class[class_id]++;
                        } else {
                            fp_for_thresh++;
                            fp_for_thresh_per_class[class_id]++;
                        }
                    }
                }

            }

        }


        unique_truth_count += nums_labels;

        //static int previous_errors = 0;
        //int total_errors = fp_for_thresh + (unique_truth_count - tp_for_thresh);
        //int errors_in_this_image = total_errors - previous_errors;
        //previous_errors = total_errors;
        //if(reinforcement_fd == NULL) reinforcement_fd = fopen("reinforcement.txt", "wb");
        //char buff[1000];
        //sprintf(buff, "%s\n", path);
        //if(errors_in_this_image > 0) fwrite(buff, sizeof(char), strlen(buff), reinforcement_fd);

//            free_detections(dets, m);
//            free(id);
//            free_image(val[t]);
//            free_image(val_resized[t]);



        //for (t = 0; t < nthreads; ++t) {
        //    pthread_join(thr[t], 0);
        //}
    }

    if ((tp_for_thresh + fp_for_thresh) > 0)
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

    int class_id;
    for (class_id = 0; class_id < classes; class_id++) {
        if ((tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]) > 0)
            avg_iou_per_class[class_id] = avg_iou_per_class[class_id] /
                                          (tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]);
    }

    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t **pr = (pr_t **) xcalloc(classes, sizeof(pr_t *));
    int i;
    for (i = 0; i < classes; ++i) {
        pr[i] = (pr_t *) xcalloc(detections_count, sizeof(pr_t));
    }

//    printf("\n detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);

    int *detection_per_class_count = (int *) xcalloc(classes, sizeof(int));
    for (int j = 0; j < detections_count; ++j) {

        detection_per_class_count[detections[j].class_id]++;
    }

    int *truth_flags = (int *) xcalloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        //        if (rank % 100 == 0)
        //            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0) {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            } else {
                pr[d.class_id][rank].fp++;
//                fprintf(fp_file, "fp: %s\n", paths[d.image_index]);
            }
        } else {
            pr[d.class_id][rank].fp++;    // false-positive
//            fprintf(fp_file, "fp: %s\n", paths[d.image_index]);
        }

        for (i = 0; i < classes; ++i) {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double) tp / (double) (tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double) tp / (double) (tp + fn);
            else pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && detection_per_class_count[i] != (tp + fp)) {    // check for last rank
                printf(" class_id: %d - detections = %7d, tp+fp = %7d, tp = %6d, fp = %6d , fn = %6d \n", i,
                       detection_per_class_count[i], tp + fp, tp, fp, fn);
            }
        }
    }

    free(truth_flags);


    double mean_average_precision = 0;

    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;

        // MS COCO - uses 101-Recall-points on PR-chart.
        // PascalVOC2007 - uses 11-Recall-points on PR-chart.
        // PascalVOC2010-2012 - uses Area-Under-Curve on PR-chart.
        // ImageNet - uses Area-Under-Curve on PR-chart.

        // correct mAP calculation: ImageNet, PascalVOC 2010-2012
        if (map_points == 0) {
            double last_recall = pr[i][detections_count - 1].recall;
            double last_precision = pr[i][detections_count - 1].precision;
            for (rank = detections_count - 2; rank >= 0; --rank) {
                double delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) {
                    last_precision = pr[i][rank].precision;
                }

                avg_precision += delta_recall * last_precision;
            }
            //add remaining area of PR curve when recall isn't 0 at rank-1
            double delta_recall = last_recall - 0;
            avg_precision += delta_recall * last_precision;
        }
            // MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
        else {
            int point;
            for (point = 0; point < map_points; ++point) {
                double cur_recall = point * 1.0 / (map_points - 1);
                double cur_precision = 0;
                for (rank = 0; rank < detections_count; ++rank) {
                    if (pr[i][rank].recall >= cur_recall) {    // > or >=
                        if (pr[i][rank].precision > cur_precision) {
                            cur_precision = pr[i][rank].precision;
                        }
                    }
                }
                //printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

                avg_precision += cur_precision;
            }
            avg_precision = avg_precision / map_points;
        }

        float class_precision = (float) tp_for_thresh_per_class[i] /
                                ((float) tp_for_thresh_per_class[i] + (float) fp_for_thresh_per_class[i]);
        float class_recall = (float) tp_for_thresh_per_class[i] /
                             ((float) tp_for_thresh_per_class[i] +
                              (float) (truth_classes_count[i] - tp_for_thresh_per_class[i]));
        //printf("Precision = %1.2f, Recall = %1.2f, avg IOU = %2.2f%% \n\n", class_precision, class_recall, avg_iou_per_class[i]);

        printf("class_id =%2d, name =%16s, count = (%7d/%7d), ap = %2.2f%%   "
               "(TP = %d, FP = %d, FN = %d, precision = %2.1f%%, recall = %2.1f%%, F1 = %2.1f%%, avg_iou = %2.1f%%) \n",
               i, names[i], detection_per_class_count[i], truth_classes_count[i], avg_precision * 100,
               tp_for_thresh_per_class[i], fp_for_thresh_per_class[i],
               truth_classes_count[i] - tp_for_thresh_per_class[i], class_precision * 100, class_recall * 100,
               2 * class_precision * class_recall / (class_precision + class_recall) * 100, avg_iou_per_class[i] * 100);


        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float) tp_for_thresh / ((float) tp_for_thresh + (float) fp_for_thresh);
    const float cur_recall =
            (float) tp_for_thresh / ((float) tp_for_thresh + (float) (unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf("\n for conf_thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
           thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for conf_thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
           thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / classes;
    printf("\n IoU threshold = %2.0f %%, ", iou_thresh * 100);
    if (map_points) printf("used %d Recall-points \n", map_points);
    else printf("used Area-Under-Curve for each unique Recall \n");

    printf(" mean average precision (mAP@%0.2f) = %f, or %2.2f %% \n", iou_thresh, mean_average_precision,
           mean_average_precision * 100);

//    for (i = 0; i < classes; ++i) {
//        free(pr[i]);
//    }
    free(pr);
    free(detections);
    free(truth_classes_count);
    free(detection_per_class_count);
//
    free(avg_iou_per_class);
    free(tp_for_thresh_per_class);
    free(fp_for_thresh_per_class);;
    if (reinforcement_fd != NULL) fclose(reinforcement_fd);


//    fclose(fp_file);
//    fclose(fn_file);
//

//    return mean_average_precision;
//    return 0;
}