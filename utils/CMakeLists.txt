cmake_minimum_required(VERSION 3.10.2)
set(PROJECT_NAME yolov3_ncnn)
project(${PROJECT_NAME} VERSION 1.1.3)

set(CMAKE_CXX_STANDARD 14)

# opencv
set(OpenCV_DIR /usr/local/share/OpenCV)
message("asd" ${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)

#ncnn

set(NCNN_LIBS "/home/fandong/Code/ncnn/build22/install/lib/libncnn.a")
link_directories(/home/fandong/Code/ncnn/build22/install/lib)
link_libraries(ncnn)

set(ncnn_DIR "/home/fandong/Code/ncnn/build22/install/lib/cmake/ncnn")
set(NCNN_INCLUDE_DIRS "/home/fandong/Code/ncnn/build22/install/include/ncnn")
include_directories(NCNN_INCLUDE_DIRS)
find_package(ncnn REQUIRED)


# 搜索本目录搜索源码并赋值给变量
aux_source_directory(. DIR_LIB_SRCS)



add_library(${PROJECT_NAME} SHARED ${DIR_LIB_SRCS})