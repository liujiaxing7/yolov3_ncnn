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

void ReadFilesFromDir(const std::string &path_to_dir
                      , std::vector<std::string> *image_name_list);

void ReadFile(std::string srcFile, std::vector<std::string> &image_files);

//void ReadArray(const YAML::Node &config, std::vector<float> &array);

//bool GetCameraConfig(std::string file, swr::imsee_types::MoudleAllParam &param);
//
//bool GetLidar(std::string yamlFile, swr::sensor_types::LaserScan &data, bool mode);
//
//bool GetPose(std::string yamlFile, swr::slam_types::SlamResult &pose);

std::string getCurrentExePath();
std::string getCurrentExeName();

#endif //DETECTOR_SAMPLE_UTILS_H
