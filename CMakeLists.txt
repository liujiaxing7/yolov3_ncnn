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


set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/)
file(GLOB file ${ROOT_DIR}/sample/*)
add_executable(${PROJECT_NAME} sample/main.cpp ${file})

# generate readme of sample for API
add_library(yolov3ncnn SHARED sample/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(
        yolov3ncnn
        ncnn

        )
#add_executable(yolov3_ncnn1 sample/main.cpp sample/utils/yolov3.cpp sample/utils/yolov3.h)
