//
// Created by wangzha on 9/28/22.
//

#include "feature_mapping.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "thinImage.h"

using namespace std;
using namespace cv;

Mat getHorizontal(Mat src){

    Mat horizontal = src.clone();
    int scale_H = 140; //这个值越大，检测到的横线越多
    int horizontalsize = horizontal.cols / scale_H;
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    return horizontal;
}

Mat getVertical(Mat src){

    Mat vertical = src.clone();
    int scale_V = 100; //这个值越大，检测到的横线越多
    int verticalsize = vertical.rows / scale_V;
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));

    return vertical;
}
Mat imagePreprocess(Mat src){
    Mat result;
    imshow("original image",src);
    medianBlur(src,result,3);
    imshow("process image",result);
    return result;
}
Mat matching_pictures(Mat src1,Mat src2){
    vector<KeyPoint> keypoints_1, keypoints_2;

    Mat descriptors_1, descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    //first detect Oriented Fast feature
    orb->detect(src1, keypoints_1);
    orb->detect(src2, keypoints_2);

    orb->compute(src1, keypoints_1, descriptors_1);
    orb->compute(src2, keypoints_2, descriptors_2);

    Mat outimg1;
    drawKeypoints(src1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);


    //matching points by using hamming
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);
    //filter points
    double min_dist = 10000, max_dist = 0;


    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist)min_dist = dist;
        if (dist > max_dist)max_dist = dist;
    }
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    //-- 第五步: 绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(src1, keypoints_1, src2, keypoints_2, matches, img_match);
    drawMatches(src1, keypoints_1, src2, keypoints_2, good_matches, img_goodmatch);
    return img_goodmatch;
}
/// <summary>
/// SIFT算法进行特征提取及匹配  10.4 添加
/// </summary>
/// <param name="image1"></param>
/// <param name="image2"></param>
void sift(Mat image1, Mat image2) {
    int64 t1=0, t2=0;
    double tkpt, tdes, tmatch_bf, tmatch_knn;

    // 1. 读取图片
    //const cv::Mat image1 = cv::imread("../../images/1.png", 0); //Load as grayscale
    //const cv::Mat image2 = cv::imread("../../images/2.png", 0); //Load as grayscale
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;

    Ptr<cv::SiftFeatureDetector> sift = cv::SiftFeatureDetector::create();
    // 2. 计算特征点
    t1 = cv::getTickCount();
    sift->detect(image1, keypoints1);
    t2 = cv::getTickCount();
    tkpt = 1000.0 * (t2 - t1) / cv::getTickFrequency();
    sift->detect(image2, keypoints2);

   //计算匹配符
    //
    Mat descriptors1;
    Mat descriptors2;
    t1 = cv::getTickCount();
    sift->compute(image1, keypoints1, descriptors1);
    t2 = cv::getTickCount();
    tdes = 1000.0 * (t2 - t1) / cv::getTickFrequency();
    sift->compute(image2, keypoints2, descriptors2);

    // 4. 特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    // cv::BFMatcher matcher(cv::NORM_L2);

    // (1) 直接暴力匹配
    std::vector<cv::DMatch> matches;
    t1 = cv::getTickCount();
    matcher->match(descriptors1, descriptors2, matches);
    t2 = cv::getTickCount();
    tmatch_bf = 1000.0 * (t2 - t1) / cv::getTickFrequency();
    // 画匹配图
    cv::Mat img_matches_bf;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches_bf);
    imshow("bf_matches", img_matches_bf);

    
    std::vector<std::vector<DMatch> > knn_matches;
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    t1 = getTickCount();
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    for (auto& knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }
    t2 = getTickCount();
    tmatch_knn = 1000.0 * (t2 - t1) / getTickFrequency();

    // 画匹配图
    cv::Mat img_matches_knn;
    drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches_knn, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("knn_matches", img_matches_knn);
    cv::waitKey(0);



    cv::Mat output;
    cv::drawKeypoints(image1, keypoints1, output);
    cv::imwrite("sift_image1_keypoints",output);
    cv::drawKeypoints(image2, keypoints2, output);
    cv::imwrite("sift_image2_keypoints", output);
    
}
int main() {

    Mat image_left = imread("/home/wangzha/Desktop/3D-constructure/resources/left.bmp", IMREAD_GRAYSCALE);
    Mat image_right = imread("/home/wangzha/Desktop/3D-constructure/resources/right.bmp", IMREAD_GRAYSCALE);

    // binary image pictures
    adaptiveThreshold(image_left, image_left, 255,ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
    adaptiveThreshold(image_right, image_right, 255,ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);




    Mat left_resize, right_resize;
    resize(image_left, left_resize, Size(image_left.rows / 4, image_left.cols / 2), INTER_LINEAR);
    resize(image_right, right_resize, Size(image_left.rows / 4, image_left.cols / 2), INTER_LINEAR);

    left_resize= imagePreprocess(left_resize);


    Mat horizontal_left = getHorizontal(left_resize);
    Mat horizontal_right = getHorizontal(right_resize);

    //细化骨架
    cv::Mat dst = thinImage(horizontal_left);
    dst = dst * 255;

    imshow("horizontal left",horizontal_left);
    imshow("thin image",dst);
//    Mat goodMatch_horizontal= matching_pictures(horizontal_left,horizontal_right);

//    imshow("优化后匹配点对", goodMatch_horizontal);

    cv::waitKey(0);
}
