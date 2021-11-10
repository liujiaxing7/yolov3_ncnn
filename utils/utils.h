//
// Created by hao on 2021/6/15.
//

#ifndef DETECTOR_SAMPLE_UTILS_H
#define DETECTOR_SAMPLE_UTILS_H
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>
#include <dirent.h>
//#include <yaml-cpp/yaml.h>
// robot
//#include "imsee_types.h"
//#include "slam_types.h"
//#include "sensor_types.h"

#define ERROR_PRINT(x) std::cout << "\033[31m" << (x) << "\033[0m" << std::endl
#define WARNING_PRINT(x) std::cout << "\033[33m" << (x) << "\033[0m" << std::endl
#define INFO_PRINT(x) std::cout << "\033[0m" << (x) << "\033[0m" << std::endl

typedef struct box_label {
    int id;
    int track_id;
    float x, y, w, h;
    float left, right, top, bottom;
} box_label;

void ReadFilesFromDir(const std::string &path_to_dir
                      , std::vector<std::string> *image_name_list);

void ReadFile(std::string srcFile, std::vector<std::string> &image_files);

box_label *read_boxes(char *filename, int *n);
void replace_image_to_label(const char* input_path, char* output_path);
void *xcalloc(size_t nmemb, size_t size);
void *xmalloc(size_t size);
void trim(char *str);
unsigned long custom_hash(char *str);
void *xrealloc(void *ptr, size_t size);
void find_replace_extension(char *str, char *orig, char *rep, char *output);
void malloc_error();
void calloc_error();
void realloc_error();
void find_replace(const char* str, char* orig, char* rep, char* output);
std::string getCurrentExePath();
std::string getCurrentExeName();


#endif //DETECTOR_SAMPLE_UTILS_H
