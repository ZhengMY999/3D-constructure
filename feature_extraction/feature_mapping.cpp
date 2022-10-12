//
// Created by wangzha on 9/28/22.
//

#include "feature_mapping.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <algorithm>


using namespace std;
using namespace cv;


Mat imagePreprocess(Mat src){
    Mat result;
    imshow("original image",src);
    medianBlur(src,result,3); //中值滤波
    imshow("process image",result);
    return result;
}
Mat matching_pictures(Mat src1,Mat src2, Mat src3, Mat src4){
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
        if (matches[i].distance <= MIN(2 * min_dist, 60.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    DMatch best_match= matches[0];
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= best_match.distance) {
            best_match = matches[i];
        }
    }
    std::vector<DMatch> best_matches;
    best_matches.push_back(best_match);

    //-- 第五步: 绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    Mat img_bestmatch;
    drawMatches(src1, keypoints_1, src2, keypoints_2, matches, img_match);
    drawMatches(src1, keypoints_1, src2, keypoints_2, good_matches, img_goodmatch);
    drawMatches(src1, keypoints_1, src2, keypoints_2, best_matches, img_bestmatch);
    return img_goodmatch;
}


// SIFT算法进行特征提取及匹配  10.4 添加

void sift(Mat image1, Mat image2,Mat image3,Mat image4) {
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

    std::vector<cv::KeyPoint> joint_keypoints1;
    std::vector<cv::KeyPoint> joint_keypoints2;
    Ptr<ORB> orb = ORB::create(400, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    orb->detect(image3, joint_keypoints1);
    orb->detect(image4, joint_keypoints2);
    //去除临近稠密特征点
    simplify_point(joint_keypoints1,30);
    simplify_point(joint_keypoints2,30);
    printf("joint_keypoints1=%d \n",joint_keypoints1.size());
    printf("joint_keypoints2=%d \n", joint_keypoints2.size());


   //计算匹配符
    //
    Mat descriptors1;
    Mat descriptors2;
    t1 = cv::getTickCount();
    sift->compute(image1, keypoints1, descriptors1);
    t2 = cv::getTickCount();
    tdes = 1000.0 * (t2 - t1) / cv::getTickFrequency();
    sift->compute(image2, keypoints2, descriptors2);
    //imshow("descriptors1", descriptors1);
    printf("descriptors compute finished\n");

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
    //drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches_bf);
    //imshow("bf_matches", img_matches_bf);

    printf("matches finished\n");

    /*
    //KNN 匹配
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
    //cv::waitKey(0);
    */


    //画出找到的特征点
    /*
    cv::Mat output;
    cv::drawKeypoints(image1, keypoints1, output);
    cv::imshow("sift_image1_keypoints",output);
    cv::drawKeypoints(image2, keypoints2, output);
    cv::imshow("sift_image2_keypoints", output);
    */


    ///以下为RANSAC消除错误匹配点

    vector<DMatch> RR_matches=RANSAC_demo(image1, image2,keypoints1, keypoints2, matches);
    

    
    
    //以下为通过已知匹配点，建立左右视图中网格点的对应关系
    // 
    //先将网格交点特征点存储在向量中

    vector<float> point_x1;
    vector<float> point_y1;
    vector<float> point_x2;
    vector<float> point_y2;
    for (int i = 0; i < joint_keypoints1.size(); i++) {
        point_x1.push_back(joint_keypoints1[i].pt.x);
        point_y1.push_back(joint_keypoints1[i].pt.y);
    }
    for (int i = 0; i < joint_keypoints2.size(); i++) {
        point_x2.push_back(joint_keypoints2[i].pt.x);
        point_y2.push_back(joint_keypoints2[i].pt.y);
    }


    //'''''''''''将强特征点周围的四个弱特征点进行匹配'''''''''''''
    //joint_keypoints3,joint_keypoints4存储强特征点周围的弱特征点

    std::vector<cv::KeyPoint> joint_keypoints3;
    std::vector<cv::KeyPoint> joint_keypoints4;
    vector<DMatch> joint_matches;
    for (int i = 0; i < RR_matches.size(); i++) {
        int x1, x2, y1, y2;
        int index1, index2;

        //'''''''''''''(x1,y1)  (x2,y2) 是相匹配的两个强特征点的坐标'''''''''''''''

        index1 = RR_matches[i].queryIdx;
        index2 = RR_matches.at(i).trainIdx;
        x1=keypoints1[RR_matches[i].queryIdx].pt.x;
        x2=keypoints2[RR_matches[i].trainIdx].pt.x;
        y1 = keypoints1[RR_matches[i].queryIdx].pt.y;
        y2 = keypoints2[RR_matches[i].trainIdx].pt.y;
        
        int nearest1= find_nearestpoint(x1,y1, point_x1, point_y1);
        
        //''''''''''''''''在图1的交点特征点中寻找与强特征点最近的特征点''''''''''''''''''

        int nearest1_x = point_x1[nearest1];
        int nearest1_y = point_y1[nearest1];
        int nearest2_x = x2 + (nearest1_x - x1);
        int nearest2_y = y2 + (nearest1_y - y1);
        int nearest2 = find_nearestpoint(nearest2_x, nearest2_y, point_x2, point_y2);
        //int nearest2 = find_nearestpoint(x2, y2, point_x2, point_y2);
        float distance= (nearest2_x - point_x2[nearest2]) * (nearest2_x - point_x2[nearest2]) + (nearest2_y - point_y2[nearest2]) * (nearest2_y - point_y2[nearest2]);//对应交点特征点的距离
        int K_distance = 10;
        if (distance < K_distance) {
            //'''''''''该值可更改'''''''''''
            KeyPoint p1;
            p1.pt.x = nearest1_x;
            p1.pt.y = nearest1_y;
                       
            KeyPoint p2;
            p2.pt.x = point_x2[nearest2];
            p2.pt.y = point_y2[nearest2];
            
            int j;
            for ( j = 0; j < joint_keypoints3.size(); j++) {   //判断是否有重复的
                if (joint_keypoints3[j].pt == p1.pt && joint_keypoints4[j].pt == p2.pt)
                    break;
            }
            
            if (j == joint_keypoints3.size()) {
                joint_keypoints3.push_back(p1);
                joint_keypoints4.push_back(p2);

                DMatch new_match(joint_keypoints3.size() - 1, joint_keypoints3.size() - 1, distance);
                joint_matches.push_back(new_match);
            }
              
        }
        
        // 寻找其他三个方向的匹配点
         int nearest11 = find_nearestpoint(x1+(nearest1_x - x1), y1 - (nearest1_y - y1), point_x1, point_y1);
         int nearest12 = find_nearestpoint(x1 - (nearest1_x - x1), y1 - (nearest1_y - y1), point_x1, point_y1);
         int nearest13 = find_nearestpoint(x1 - (nearest1_x - x1), y1 + (nearest1_y - y1), point_x1, point_y1);

         int nearest11_x = point_x1[nearest11];
         int nearest11_y = point_y1[nearest11];
         int nearest21_x = x2 + (nearest11_x - x1);
         int nearest21_y = y2 + (nearest11_y - y1);
         int nearest21 = find_nearestpoint(nearest21_x, nearest21_y, point_x2, point_y2);

         int nearest12_x = point_x1[nearest12];
         int nearest12_y = point_y1[nearest12];
         int nearest22_x = x2 + (nearest12_x - x1);
         int nearest22_y = y2 + (nearest12_y - y1);
         int nearest22 = find_nearestpoint(nearest22_x, nearest22_y, point_x2, point_y2);

         int nearest13_x = point_x1[nearest13];
         int nearest13_y = point_y1[nearest13];
         int nearest23_x = x2 + (nearest13_x - x1);
         int nearest23_y = y2 + (nearest13_y - y1);
         int nearest23 = find_nearestpoint(nearest23_x, nearest23_y, point_x2, point_y2);

         int distance1 = (nearest21_x - point_x2[nearest21]) * (nearest21_x - point_x2[nearest21]) + (nearest21_y - point_y2[nearest21]) * (nearest21_y - point_y2[nearest21]);//对应交点特征点的距离
         int distance2 = (nearest22_x - point_x2[nearest22]) * (nearest22_x - point_x2[nearest22]) + (nearest22_y - point_y2[nearest22]) * (nearest22_y - point_y2[nearest22]);//对应交点特征点的距离
         int distance3 = (nearest23_x - point_x2[nearest23]) * (nearest23_x - point_x2[nearest23]) + (nearest23_y - point_y2[nearest23]) * (nearest23_y - point_y2[nearest23]);//对应交点特征点的距离

        if (distance < K_distance) {
            //'''''''''该值可更改'''''''''''
            KeyPoint p1;
            p1.pt.x = nearest11_x;
            p1.pt.y = nearest11_y;

            KeyPoint p2;
            p2.pt.x = point_x2[nearest21];
            p2.pt.y = point_y2[nearest21];

            int j;
            for (j = 0; j < joint_keypoints3.size(); j++) {   //判断是否有重复的
                if (joint_keypoints3[j].pt == p1.pt && joint_keypoints4[j].pt == p2.pt)
                    break;
            }

            if (j == joint_keypoints3.size()) {
                joint_keypoints3.push_back(p1);
                joint_keypoints4.push_back(p2);

                DMatch new_match(joint_keypoints3.size() - 1, joint_keypoints3.size() - 1, distance1);
                joint_matches.push_back(new_match);
            }
        }
        if (distance < K_distance) {
            //'''''''''该值可更改'''''''''''
            KeyPoint p1;
            p1.pt.x = nearest12_x;
            p1.pt.y = nearest12_y;

            KeyPoint p2;
            p2.pt.x = point_x2[nearest22];
            p2.pt.y = point_y2[nearest22];

            int j;
            for (j = 0; j < joint_keypoints3.size(); j++) {   //判断是否有重复的
                if (joint_keypoints3[j].pt == p1.pt && joint_keypoints4[j].pt == p2.pt)
                    break;
            }

            if (j == joint_keypoints3.size()) {
                joint_keypoints3.push_back(p1);
                joint_keypoints4.push_back(p2);

                DMatch new_match(joint_keypoints3.size() - 1, joint_keypoints3.size() - 1, distance1);
                joint_matches.push_back(new_match);
            }
        }
        if (distance < K_distance) {
            //'''''''''该值可更改'''''''''''
            KeyPoint p1;
            p1.pt.x = nearest13_x;
            p1.pt.y = nearest13_y;

            KeyPoint p2;
            p2.pt.x = point_x2[nearest23];
            p2.pt.y = point_y2[nearest23];

            int j;
            for (j = 0; j < joint_keypoints3.size(); j++) {   //判断是否有重复的
                if (joint_keypoints3[j].pt == p1.pt && joint_keypoints4[j].pt == p2.pt)
                    break;
            }

            if (j == joint_keypoints3.size()) {
                joint_keypoints3.push_back(p1);
                joint_keypoints4.push_back(p2);

                DMatch new_match(joint_keypoints3.size() - 1, joint_keypoints3.size() - 1, distance1);
                joint_matches.push_back(new_match);
            }
        }

    }
    
    
    //cv::Mat joint_match;
    //drawMatches(image3, joint_keypoints3, image4, joint_keypoints4, joint_matches, joint_match);
    //imshow("joint_match", joint_match);
    
    RR_matches = RANSAC_demo(image3, image4, joint_keypoints3, joint_keypoints4, joint_matches);

    
    //将匹配特征点从网格特征点中去除
    for (int i = 0; i < joint_keypoints3.size(); i++) {
        int index = vector_find(joint_keypoints1, joint_keypoints3[i]) ;
        if (index != joint_keypoints1.size()) {
            vector<KeyPoint>::iterator it = index + joint_keypoints1.begin();
            joint_keypoints1.erase(it);
        }
    }
    for (int i = 0; i < joint_keypoints4.size(); i++) {
        int index = vector_find(joint_keypoints2, joint_keypoints4[i]);
        if (index != joint_keypoints2.size()) {
            vector<KeyPoint>::iterator it = index + joint_keypoints2.begin();
            joint_keypoints2.erase(it);
        }
    }

    
  
    weakFeature_mapping(RR_matches, joint_keypoints3, joint_keypoints4, joint_keypoints1, joint_keypoints2);
    printf("joint_keypoints1  %d\n", joint_keypoints1.size());
    int last_point_number= joint_keypoints1.size()+10;
    while (1) {   //重复以上过程，将强特征点附近的弱特征点进行匹配
        weakFeature_mapping(RR_matches, joint_keypoints3, joint_keypoints4, joint_keypoints1, joint_keypoints2);
        printf("joint_keypoints1  %d\n", joint_keypoints1.size());
        RR_matches = RANSAC_demo(image3, image4, joint_keypoints3, joint_keypoints4, RR_matches);
        //
        if (last_point_number == joint_keypoints1.size())    //强特征点无法增加时，结束
            break;
        last_point_number = joint_keypoints1.size();
    }
    RR_matches = RANSAC_demo(image3, image4, joint_keypoints3, joint_keypoints4, RR_matches);

    cv::Mat output;
    cv::drawKeypoints(image3, joint_keypoints1, output);
    cv::imshow("joint_keypoints1", output);


    cv::drawKeypoints(image3, joint_keypoints3, output);
    cv::imshow("joint_keypoints3", output);
    cv::drawKeypoints(image4, joint_keypoints4, output);
    cv::imshow("joint_keypoints4", output);
    //需要过滤极度临近的特征点
}
/// <summary>
/// RANSAC消除错误匹配点
/// </summary>
/// <returns></returns>
std::vector<DMatch> RANSAC_demo(Mat image1,Mat image2, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<DMatch> good_matches) {
    vector<KeyPoint> R_keypoint01, R_keypoint02;
    for (size_t i = 0; i < good_matches.size(); i++) {
        R_keypoint01.push_back(keypoints1[good_matches[i].queryIdx]);
        R_keypoint02.push_back(keypoints2[good_matches[i].trainIdx]);

        //这两句话的理解：R_keypoint1是要存储img01中能与img02匹配的特征点
        //matches中存储了这些匹配点对的img01和img02的索引值

    }
    //坐标转换
    vector<Point2f>p01, p02;
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }

    //利用基础矩阵剔除误匹配点
    vector<uchar> RansacStatus;
    Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);


    vector<KeyPoint> RR_keypoint01, RR_keypoint02;
    vector<DMatch> RR_matches;

    //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵

    int index = 0;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        if (RansacStatus[i] != 0)
        {
            RR_keypoint01.push_back(R_keypoint01[i]);
            RR_keypoint02.push_back(R_keypoint02[i]);
            good_matches[i].queryIdx = index;
            good_matches[i].trainIdx = index;
            RR_matches.push_back(good_matches[i]);
            index++;
        }
    }
    Mat img_RR_matches;
    drawMatches(image1, RR_keypoint01, image2, RR_keypoint02, RR_matches, img_RR_matches);
    imshow("img_RR_matches", img_RR_matches);
    
    keypoints1 = RR_keypoint01;
    keypoints2 = RR_keypoint02;
    printf("RR_matches.size() %d\n", RR_matches.size());
    return RR_matches;
}

int find_nearestpoint(int x, int y, std::vector<float> point_x, std::vector<float> point_y) {
    int nearest = 0;
    float distance = (point_x[0] - x) * (point_x[0] - x) + (point_y[0] - y) * (point_y[0] - y);
    for (int i = 0; i < point_x.size(); i++) {
        if ((point_x[i] - x) * (point_x[i] - x) + (point_y[i] - y) * (point_y[i] - y) < distance) {
            distance = (point_x[i] - x) * (point_x[i] - x) + (point_y[i] - y) * (point_y[i] - y);
            nearest = i;
        }
    }
    return nearest;
}
int find_nearestpoint(int x, int y, std::vector<cv::KeyPoint> joint_keypoints) {
    int nearest = 0;
    float distance = (joint_keypoints[0].pt.x - x) * (joint_keypoints[0].pt.x - x) + (joint_keypoints[0].pt.y - y) * (joint_keypoints[0].pt.y - y);
    for (int i = 0; i < joint_keypoints.size(); i++) {
        if ((joint_keypoints[i].pt.x - x) * (joint_keypoints[i].pt.x - x) + (joint_keypoints[i].pt.y - y) * (joint_keypoints[i].pt.y - y) < distance) {
            distance = (joint_keypoints[i].pt.x - x) * (joint_keypoints[i].pt.x - x) + (joint_keypoints[i].pt.y - y) * (joint_keypoints[i].pt.y - y);
            nearest = i;
        }
    }
    return nearest;
}

/// <summary>
/// 根据已知强特征点匹配，进行弱特征点匹配
/// </summary>
/// <param name="RR_matches 已知强特征点匹配关系 "></param>
/// <param name="keypoints1 keypoints2 已知强特征点 "></param>
/// <param name="joint_keypoints1 joint_keypoints2 剩余网格特征点，即弱特征点 "></param>
void weakFeature_mapping(std::vector<DMatch> &RR_matches,std::vector<cv::KeyPoint> &keypoints1,std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &joint_keypoints1, std::vector<cv::KeyPoint> &joint_keypoints2) {
    //先将网格交点特征点存储在向量中

    vector<float> point_x1;
    vector<float> point_y1;
    vector<float> point_x2;
    vector<float> point_y2;
    for (int i = 0; i < joint_keypoints1.size(); i++) {
        point_x1.push_back(joint_keypoints1[i].pt.x);
        point_y1.push_back(joint_keypoints1[i].pt.y);
    }
    for (int i = 0; i < joint_keypoints2.size(); i++) {
        point_x2.push_back(joint_keypoints2[i].pt.x);
        point_y2.push_back(joint_keypoints2[i].pt.y);
    }
    //'''''''''''将强特征点周围的四个弱特征点进行匹配'''''''''''''
   //joint_keypoints3,joint_keypoints4存储强特征点周围的弱特征点

    //std::vector<cv::KeyPoint> joint_keypoints3;
    //std::vector<cv::KeyPoint> joint_keypoints4;
    //vector<DMatch> joint_matches;
    for (int i = 0; i < RR_matches.size(); i++) {
        int x1, x2, y1, y2;
        int index1, index2;

        //'''''''''''''(x1,y1)  (x2,y2) 是相匹配的两个强特征点的坐标'''''''''''''''

        index1 = RR_matches[i].queryIdx;
        index2 = RR_matches.at(i).trainIdx;
        x1 = keypoints1[RR_matches[i].queryIdx].pt.x;
        x2 = keypoints2[RR_matches[i].trainIdx].pt.x;
        y1 = keypoints1[RR_matches[i].queryIdx].pt.y;
        y2 = keypoints2[RR_matches[i].trainIdx].pt.y;

        int nearest1 = find_nearestpoint(x1, y1, point_x1, point_y1);

        //''''''''''''''''在图1的交点特征点中寻找与强特征点最近的特征点''''''''''''''''''

        int nearest1_x = point_x1[nearest1];
        int nearest1_y = point_y1[nearest1];
        int nearest2_x = x2 + (nearest1_x - x1);
        int nearest2_y = y2 + (nearest1_y - y1);
        int nearest2 = find_nearestpoint(nearest2_x, nearest2_y, point_x2, point_y2);
        //int nearest2 = find_nearestpoint(x2, y2, point_x2, point_y2);
        float distance = (nearest2_x - point_x2[nearest2]) * (nearest2_x - point_x2[nearest2]) + (nearest2_y - point_y2[nearest2]) * (nearest2_y - point_y2[nearest2]);//对应交点特征点的距离
        int K_distance = 10;
        if (distance < K_distance) {

            //'''''''''该值可更改'''''''''''
        KeyPoint p1;
        p1.pt.x = nearest1_x;
        p1.pt.y = nearest1_y;

        KeyPoint p2;
        p2.pt.x = point_x2[nearest2];
        p2.pt.y = point_y2[nearest2];

        int j;
        for (j = 0; j < keypoints1.size(); j++) {   //判断是否有重复的
            if (keypoints1[j].pt == p1.pt && keypoints2[j].pt == p2.pt)
                break;
        }

        if (j == keypoints1.size()) {
            keypoints1.push_back(p1);
            keypoints2.push_back(p2);

            DMatch new_match(keypoints1.size() - 1, keypoints1.size() - 1, distance);
            RR_matches.push_back(new_match);
        }

        }

        // 寻找其他三个方向的匹配点
        int nearest11 = find_nearestpoint(x1 + (nearest1_x - x1), y1 - (nearest1_y - y1), point_x1, point_y1);
        int nearest12 = find_nearestpoint(x1 - (nearest1_x - x1), y1 - (nearest1_y - y1), point_x1, point_y1);
        int nearest13 = find_nearestpoint(x1 - (nearest1_x - x1), y1 + (nearest1_y - y1), point_x1, point_y1);

        int nearest11_x = point_x1[nearest11];
        int nearest11_y = point_y1[nearest11];
        int nearest21_x = x2 + (nearest11_x - x1);
        int nearest21_y = y2 + (nearest11_y - y1);
        int nearest21 = find_nearestpoint(nearest21_x, nearest21_y, point_x2, point_y2);

        int nearest12_x = point_x1[nearest12];
        int nearest12_y = point_y1[nearest12];
        int nearest22_x = x2 + (nearest12_x - x1);
        int nearest22_y = y2 + (nearest12_y - y1);
        int nearest22 = find_nearestpoint(nearest22_x, nearest22_y, point_x2, point_y2);

        int nearest13_x = point_x1[nearest13];
        int nearest13_y = point_y1[nearest13];
        int nearest23_x = x2 + (nearest13_x - x1);
        int nearest23_y = y2 + (nearest13_y - y1);
        int nearest23 = find_nearestpoint(nearest23_x, nearest23_y, point_x2, point_y2);

        int distance1 = (nearest21_x - point_x2[nearest21]) * (nearest21_x - point_x2[nearest21]) + (nearest21_y - point_y2[nearest21]) * (nearest21_y - point_y2[nearest21]);//对应交点特征点的距离
        int distance2 = (nearest22_x - point_x2[nearest22]) * (nearest22_x - point_x2[nearest22]) + (nearest22_y - point_y2[nearest22]) * (nearest22_y - point_y2[nearest22]);//对应交点特征点的距离
        int distance3 = (nearest23_x - point_x2[nearest23]) * (nearest23_x - point_x2[nearest23]) + (nearest23_y - point_y2[nearest23]) * (nearest23_y - point_y2[nearest23]);//对应交点特征点的距离

        if (distance < K_distance) {
            //'''''''''该值可更改'''''''''''
            KeyPoint p1;
            p1.pt.x = nearest11_x;
            p1.pt.y = nearest11_y;

            KeyPoint p2;
            p2.pt.x = point_x2[nearest21];
            p2.pt.y = point_y2[nearest21];

            int j;
            for (j = 0; j < keypoints1.size(); j++) {   //判断是否有重复的
                if (keypoints1[j].pt == p1.pt && keypoints2[j].pt == p2.pt)
                    break;
            }

            if (j == keypoints1.size()) {
                keypoints1.push_back(p1);
                keypoints2.push_back(p2);

                DMatch new_match(keypoints1.size() - 1, keypoints1.size() - 1, distance1);
                RR_matches.push_back(new_match);
            }
        }
        if (distance < K_distance) {
            //'''''''''该值可更改'''''''''''
            KeyPoint p1;
            p1.pt.x = nearest12_x;
            p1.pt.y = nearest12_y;

            KeyPoint p2;
            p2.pt.x = point_x2[nearest22];
            p2.pt.y = point_y2[nearest22];

            int j;
            for (j = 0; j < keypoints1.size(); j++) {   //判断是否有重复的
                if (keypoints1[j].pt == p1.pt && keypoints2[j].pt == p2.pt)
                    break;
            }

            if (j == keypoints1.size()) {
                keypoints1.push_back(p1);
                keypoints2.push_back(p2);

                DMatch new_match(keypoints1.size() - 1, keypoints1.size() - 1, distance1);
                RR_matches.push_back(new_match);
            }
        }
        if (distance < K_distance) {
            //'''''''''该值可更改'''''''''''
            KeyPoint p1;
            p1.pt.x = nearest13_x;
            p1.pt.y = nearest13_y;

            KeyPoint p2;
            p2.pt.x = point_x2[nearest23];
            p2.pt.y = point_y2[nearest23];

            int j;
            for (j = 0; j < keypoints1.size(); j++) {   //判断是否有重复的
                if (keypoints1[j].pt == p1.pt && keypoints2[j].pt == p2.pt)
                    break;
            }

            if (j == keypoints1.size()) {
                keypoints1.push_back(p1);
                keypoints2.push_back(p2);

                DMatch new_match(keypoints1.size() - 1, keypoints1.size() - 1, distance1);
                RR_matches.push_back(new_match);
            }
        }

    }

    //将匹配特征点从网格特征点中去除
    for (int i = 0; i < keypoints1.size(); i++) {
        int index = vector_find(joint_keypoints1, keypoints1[i]);
        if (index != joint_keypoints1.size()) {
            vector<KeyPoint>::iterator it = index + joint_keypoints1.begin();
            joint_keypoints1.erase(it);
        }
    }
    for (int i = 0; i < keypoints2.size(); i++) {
        int index = vector_find(joint_keypoints2, keypoints2[i]);
        if (index != joint_keypoints2.size()) {
            vector<KeyPoint>::iterator it = index + joint_keypoints2.begin();
            joint_keypoints2.erase(it);
        }
    }
}

int vector_find(std::vector<cv::KeyPoint> points, cv::KeyPoint point) {
    for (int i = 0; i < points.size(); i++) {
        if (points[i].pt == point.pt)
            return i;
    }
    return points.size();
}
/// <summary>
/// 对图像进行极线校正
/// </summary>
/// <param name="src1"></param>
/// <param name="src2"></param>
/// <param name="dst1"></param>
/// <param name="dst2"></param>
void Undistort(Mat src1, Mat src2, Mat &dst1, Mat &dst2) {
    //相机内外参数
    Mat matK1 = (Mat_<float>(3, 3) << 2456.38, 0, 1812.75, 0, 2453.20, 1361.87, 0, 0, 1);//左相机相机内参矩阵
    Mat matK2 = (Mat_<float>(3, 3) << 2475.906, 0, 1855.9, 0, 2470.32, 1330.98, 0, 0, 1);//右相机相机内参矩阵
    Mat matD1 = (Mat_<float>(4, 1) << -0.25230035, 0.170470388, 0.000998086, -0.0002442);//左相机畸变向量
    Mat matD2 = (Mat_<float>(4, 1) << -0.243125154959494, 0.117156083, -0.000445847, 0.00133389);//右相机畸变向量
    Mat R12 = (Mat_<float>(3, 3) << 0.999755674, 3.94944E-05, -0.022104107, -0.000152744, 0.999986872, -0.005121818, 0.022103615, 0.005123943, 0.999742555);//相机之间的旋转矩阵
    Mat T12 = (Mat_<float>(3, 1) << -94.26874066, -0.608688519, 1.880801234);//相机之间的平移矩阵
    Mat R1, R2, P1, P2, Q, map1_1, map1_2, map2_1, map2_2;
    cv::Mat merge;


    //极线校正
    cv::fisheye::stereoRectify(matK1, matD1, matK2, matD2, src1.size(), R12, T12, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);
    cv::fisheye::initUndistortRectifyMap(matK1, matD1, R1, P1, src1.size(), CV_32F, map1_1, map1_2);
    cv::fisheye::initUndistortRectifyMap(matK2, matD2, R2, P2, src1.size(), CV_32F, map2_1, map2_2);
    cv::remap(src1, src1, map1_1, map1_2, 0);
    cv::remap(src2, src2, map2_1, map2_2, 0);

    cv::hconcat(src1, src2, merge); //图像拼接
    //imwrite("../merge.jpg", merge);
    //resize(src1, src1, Size(src1.cols / 4, src1.rows / 4), 0, 0, INTER_NEAREST); //缩小图片  
    //resize(src2, src2, Size(src2.cols / 4, src2.rows / 4), 0, 0, INTER_NEAREST); //缩小图片  
    resize(merge, merge, Size(merge.cols / 4, merge.rows / 4), 0, 0, INTER_NEAREST); //缩小图片
    //imshow("src1", src1);
    //imshow("src2", src2);
    //imshow("merge", merge);
    //imwrite("../src1.jpg", src1);
    //imwrite("../src2.jpg", src2);

    dst1 = src1;
    dst2 = src2;
}
/// <summary>
/// 简化稠密区域特征点
/// </summary>
/// <param name="joint_keypoints"></param>
/// <param name="threshold  两点之间距离小于该阈值，则被简化为一点"></param>
void simplify_point(std::vector<cv::KeyPoint> &joint_keypoints,float threshold) {
    for (int i = 0; i < joint_keypoints.size()-2; i++) {
        std::vector<cv::KeyPoint> points(joint_keypoints.begin()+i+1, joint_keypoints.end());
        int n= find_nearestpoint(joint_keypoints[i].pt.x, joint_keypoints[i].pt.y, points) + i + 1;
        //printf("n=%d \n",n);
        //printf("i=%d \n", i);
        if (n <joint_keypoints.size() && (sqrt(joint_keypoints[i].pt.x - joint_keypoints[n].pt.x) + sqrt(joint_keypoints[i].pt.y - joint_keypoints[n].pt.y) )< threshold) {
            //joint_keypoints[i].pt.x = (joint_keypoints[i].pt.x + joint_keypoints[n].pt.x) / 2;
            //joint_keypoints[i].pt.y = (joint_keypoints[i].pt.y + joint_keypoints[n].pt.y) / 2;
            //删除稠密点
            auto iter = joint_keypoints.begin()+n;
            joint_keypoints.erase(iter);
            i--;
        }   
        //printf("joint_keypoints.size()=%d \n", joint_keypoints.size());
    }
}