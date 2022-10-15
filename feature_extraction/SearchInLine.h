
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include "opencv2/calib3d/calib3d.hpp"
using namespace cv;
#ifndef SEARCHINLINE_H
#define SEARCHINLINE_H

void line_search(Mat image_L, Mat image_R, Mat point_L, Mat point_R, Mat horizontal_L, Mat vertical_L, Mat horizontal_R, Mat vertical_R);
void verticalline_search(std::vector<cv::DMatch>& matches, Mat vertical_L, Mat vertical_R, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::KeyPoint> joint_keypointsL, std::vector<cv::KeyPoint> joint_keypointsR);
void verticalline_search_func(KeyPoint keypoint, Mat& vertical, std::vector<float>& length, std::vector<cv::KeyPoint>& find_keypoint, std::vector<cv::KeyPoint> joint_keypoints, KeyPoint keypoint2, Mat& vertical2, std::vector<float>& length2, std::vector<cv::KeyPoint>& find_keypoint2, std::vector<cv::KeyPoint> joint_keypoints2);
void horizontalline_search(std::vector<DMatch>& matches, Mat horizontal_L, Mat horizontal_R, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::KeyPoint> joint_keypointsL, std::vector<cv::KeyPoint> joint_keypointsR);
void horizontalline_search_func(KeyPoint keypoint, Mat& horizontal, std::vector<float>& length, std::vector<cv::KeyPoint>& find_keypoint, std::vector<cv::KeyPoint> joint_keypoints, KeyPoint keypoint2, Mat& horizontal2, std::vector<float>& length2, std::vector<cv::KeyPoint>& find_keypoint2, std::vector<cv::KeyPoint> joint_keypoints2);
bool is_endpoint(int x, int y, Mat image);
#endif 