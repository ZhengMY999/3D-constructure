//
// Created by wangzha on 9/28/22.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include "opencv2/calib3d/calib3d.hpp"
using namespace cv;
#ifndef FEATURE_MAPPING_FEATURE_MAPPING_H
#define FEATURE_MAPPING_FEATURE_MAPPING_H

Mat matching_pictures(Mat src1,Mat src2,Mat src3, Mat src4);
void sift(Mat image1, Mat image2,Mat image3, Mat image4);
std::vector<DMatch> RANSAC_demo(Mat image1, Mat image2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<DMatch> good_matches);
int find_nearestpoint(int x, int y, std::vector<float> point_x, std::vector<float> point_y);
int find_nearestpoint(int x, int y, std::vector<cv::KeyPoint> joint_keypoints);
void weakFeature_mapping(std::vector<DMatch> &RR_matches, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &joint_keypoints1, std::vector<cv::KeyPoint> &joint_keypoints2);
int vector_find(std::vector<cv::KeyPoint> points, cv::KeyPoint point);
void Undistort(Mat src1, Mat src2, Mat& dst1, Mat& dst2);
void simplify_point(std::vector<cv::KeyPoint> &joint_keypoints, float threshold = 1);
#endif //FEATURE_MAPPING_FEATURE_MAPPING_H
