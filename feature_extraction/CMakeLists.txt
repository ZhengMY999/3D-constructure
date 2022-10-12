cmake_minimum_required( VERSION 3.0 )
project( feature_extraction )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

#set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv4)
#set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv4/opencv2)
#set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv4/opencv2/include)

set(OpenCV_DIR "/home/wangzha/Downloads/opencv/build")
find_package( OpenCV 4.0 REQUIRED ) # use this if in OpenCV2

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        "/usr/include/eigen3/"
)

add_executable( feature_extraction  main.cpp feature_mapping.h feature_mapping.cpp feature_extraction.h feature_extraction.cpp)
target_link_libraries( feature_extraction ${OpenCV_LIBS} )


