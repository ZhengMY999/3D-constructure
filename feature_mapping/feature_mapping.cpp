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
int main(){

    Mat image_left = imread("/home/wangzha/Desktop/3D-constructure/resources/left.bmp", IMREAD_GRAYSCALE);
    Mat image_right = imread("/home/wangzha/Desktop/3D-constructure/resources/right.bmp", IMREAD_GRAYSCALE);

    //adaptiveThreshold(image_left, image_left, 255,
                      //ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
    //adaptiveThreshold(image_right, image_right, 255,
                      //ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
    int down_width = 768;
    int down_height = 768;
    Mat left_resize,right_resize;
    resize(image_left, left_resize, Size(image_left.rows/2, image_left.cols/2), INTER_LINEAR);
    resize(image_right, right_resize, Size(image_left.rows/2, image_left.cols/2), INTER_LINEAR);


    //initialize
    vector<KeyPoint> keypoints_1,keypoints_2;

    Mat descriptors_1,descriptors_2;
    Ptr<ORB> orb = ORB::create(1000,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);

    //first detect Oriented Fast feature
    orb->detect(left_resize,keypoints_1);
    orb->detect(right_resize,keypoints_2);

    orb->compute(left_resize,keypoints_1,descriptors_1);
    orb->compute(right_resize,keypoints_2,descriptors_2);

    Mat outimg1;
    drawKeypoints( left_resize, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );


    //matching points by using hamming
    vector<DMatch> matches;
    BFMatcher matcher ( NORM_HAMMING );
    matcher.match ( descriptors_1, descriptors_2, matches );
    //filter points
    double min_dist=10000,max_dist=0;


    for(int i=0;i<descriptors_1.rows;i++){
        double dist = matches[i].distance;
        if(dist<min_dist)min_dist=dist;
        if(dist>max_dist)max_dist=dist;
    }
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ ){
        if( matches[i].distance <= max ( 2*min_dist, 30.0 ) ){
            good_matches.push_back ( matches[i] );
           }
    }

    //-- 第五步: 绘制匹配结果
   Mat img_match;
   Mat img_goodmatch;
   drawMatches ( left_resize, keypoints_1, right_resize, keypoints_2, matches,img_match);
  drawMatches ( left_resize, keypoints_1, right_resize, keypoints_2, good_matches, img_goodmatch );
    imshow ( "所有匹配点对", img_match );
   imshow ( "优化后匹配点对", img_goodmatch );
    waitKey(0);

    imshow("left image",left_resize);
    imshow("right image",right_resize);
    cv::waitKey(0);
}