cmake_minimum_required( VERSION 3.0 )
project( feature_mapping )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        "/usr/include/eigen3/"
)

add_executable( feature_mapping feature_mapping.cpp thinImage.cpp thinImage.h)
target_link_libraries( feature_mapping ${OpenCV_LIBS} )


