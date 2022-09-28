//
// Created by wangzha on 9/28/22.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
using namespace cv;
#ifndef FEATURE_MAPPING_FEATURE_MAPPING_H
#define FEATURE_MAPPING_FEATURE_MAPPING_H


Mat getHorizontal(Mat src);

Mat getVertical(Mat src);
Mat matching_pictures(Mat src1,Mat src2);

#endif //FEATURE_MAPPING_FEATURE_MAPPING_H
