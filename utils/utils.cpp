//
// Created by hao on 2021/6/15.
//

#include "utils.h"
#include <unistd.h>
#include "fstream"
//const swr::imsee_types::Resolution RESOLUTION = swr::imsee_types::Resolution::RES_640X400;

void ReadFilesFromDir(const std::string &path_to_dir
                      , std::vector<std::string> *image_name_list)
{
    DIR *dir;
    dir = opendir(path_to_dir.c_str());
    struct dirent *ent;
    // CHECK_NOTNULL(dir);
    while ((ent = readdir(dir)) != nullptr)
    {
        auto name = std::string(ent->d_name);
        // ignore "." ".."
        if (name.size() < 4)
        {
            continue;
        }
        auto suffix = name.substr(name.size() - 4, 4);
        if (suffix == ".png" || suffix == ".jpg")
        {
            // filter image
            image_name_list->emplace_back(name);
        }
    }

    closedir(dir);
}

void ReadFile(std::string srcFile, std::vector<std::string> &image_files)
{
    if (not access(srcFile.c_str(), 0) == 0)
    {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open())
    {
        ERROR_PRINT("read file error (" + srcFile + ")");
        exit(0);
    }

    std::string s;
    while (getline(fin, s))
    {
        image_files.push_back(s);
    }

    fin.close();
}

//void ReadArray(const YAML::Node &config, std::vector<float> &array)
//{
//    try
//    {
//        array = config.as<std::vector<float>>();
//    }
//    catch (...)
//    {
//        for (YAML::const_iterator it = config.begin(); it != config.end(); ++it)
//        {
//            array.push_back((*it).as<float>());
//        }
//    }
//}
//
//bool GetCameraConfig(std::string file, swr::imsee_types::MoudleAllParam &param)
//{
//    cv::FileStorage fileStream = cv::FileStorage(file, cv::FileStorage::READ);
//
//    if (not fileStream.isOpened())
//    {
//        std::cout << "file not exist <" + file + ">" << std::endl;
//        return false;
//    }
//
//    // TODO : the exception for lack option
//    cv::Mat_<double> k, d, p, r;
//    fileStream["Kl"] >> k;
//    fileStream["Dl"] >> d;
//    fileStream["Pl"] >> p;
//    fileStream["Rl"] >> r;
//
//    memcpy(param._left_camera[RESOLUTION]._K, k.data
//           , sizeof(param._left_camera[RESOLUTION]._K));
//    memcpy(param._left_camera[RESOLUTION]._R, r.data
//           , sizeof(param._left_camera[RESOLUTION]._R));
//    memcpy(param._left_camera[RESOLUTION]._P, p.data
//           , sizeof(param._left_camera[RESOLUTION]._P));
//    memcpy(param._left_camera[RESOLUTION]._D, d.data
//           , sizeof(param._left_camera[RESOLUTION]._D));
//
//    fileStream.release();
//
//    return true;
//}
//
//bool GetLidar(std::string yamlFile, swr::sensor_types::LaserScan &data, bool mode)
//{
//    try
//    {
//        YAML::Node config;
//
//        if (not access(yamlFile.c_str(), 0) == 0)
//        {
//            std::cout << "file not exist <" + yamlFile + ">" << std::endl;
//        }
//        config = YAML::LoadFile(yamlFile);
//
//        data.header.seq = config["Header"]["seq"].as<int>();
//        data.header.stamp = config["Header"]["stamp"].as<unsigned long long>();
//
//        data.angle_min = config["angle_min"].as<float>();
//        data.angle_max = config["angle_max"].as<float>();
//        data.angle_increment = config["angle_increment"].as<float>();
//
//        data.scan_time = config["scan_time"].as<float>();
//        data.range_min = config["range_min"].as<float>();
//        data.range_max = config["range_max"].as<float>();
//
//        // TODO : use try catch, not mode
//        if (mode)
//        {
//            data.ranges = config["ranges"].as<std::vector<float>>();
//            data.intensities = config["intensities"].as<std::vector<float>>();
//        }
//        else
//        {
//            ReadArray(config["ranges"], data.ranges);
//            ReadArray(config["intensities"], data.intensities);
//        }
//
//        return true;
//    }
//    catch (...)
//    {
//        return false;
//    }
//}
//
//bool GetPose(std::string yamlFile, swr::slam_types::SlamResult &pose)
//{
//    try
//    {
//        YAML::Node config;
//
//        if (not access(yamlFile.c_str(), 0) == 0)
//        {
//            std::cout << "file not exist <" + yamlFile + ">" << std::endl;
//        }
//        config = YAML::LoadFile(yamlFile);
//
//        pose.s_time = config["time"].as<float>();
//        pose.s_state = config["state"].as<float>();
//
//        size_t size = 0;
//        std::vector<double> position = config["position"].as<std::vector<double>>();
//        size = position.size();
//        for (size_t i = 0; i < size; ++i)
//        {
//            pose.s_position[i] = position[i];
//        }
//
//        std::vector<double> rotation = config["rotation"].as<std::vector<double>>();
//        size = rotation.size();
//        for (size_t i = 0; i < size; ++i)
//        {
//            pose.s_rotation[i] = rotation[i];
//        }
//
//        return true;
//    }
//    catch (...)
//    {
//        return false;
//    }
//}


std::string getCurrentExePath()
{
    char szPath[256] = {0};
    int ret = readlink("/proc/self/exe", szPath, sizeof(szPath) - 1);
    std::string path(szPath, ret);
    size_t last_ = path.find_last_of('/');
    if (std::string::npos != last_)
    {
        return path.substr(0, last_ + 1);
    }
    return path;
}

std::string getCurrentExeName()
{
    char szPath[256] = {0};
    int ret = readlink("/proc/self/exe", szPath, sizeof(szPath) - 1);
    std::string path(szPath, ret);
    size_t last_ = path.find_last_of('/');
    if (std::string::npos != last_)
    {
        return path.substr(last_ + 1);
    }
    return path;
}
