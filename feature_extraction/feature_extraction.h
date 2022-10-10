#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
using namespace cv;
#ifndef FEATURE_EXTRACTION_FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_FEATURE_EXTRACTION_H

Mat close_operation(Mat src, int method, int ksize);
Mat get_joints(Mat src, int scale_H, int scale_V);
static void color_retransfer_with_merge(cv::Mat& output, std::vector<cv::Mat>& chls);
static void color_transfer_with_spilt(cv::Mat& input, std::vector<cv::Mat>& chls);
cv::Mat clahe_deal(cv::Mat& src);
Mat mask(Mat src, Mat maskimage);
void cvHilditchThin1(cv::Mat& src, cv::Mat& dst);
Mat preprocess(Mat src1, Mat src2, int Ksize, int H);

#endif 