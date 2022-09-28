//
// Created by wangzha on 9/28/22.
//

#include "feature_mapping.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

#include <iostream>

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


    Mat horizontal_left = getVertical(left_resize);
    Mat horizontal_right = getVertical(right_resize);


    Mat goodMatch_horizontal= matching_pictures(horizontal_left,horizontal_right);

    imshow("优化后匹配点对", goodMatch_horizontal);

    cv::waitKey(0);
}