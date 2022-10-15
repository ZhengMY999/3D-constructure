//
// Created by wangzha on 9/28/22.
//

#include "SearchInLine.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <algorithm>
#include "feature_mapping.h"

using namespace std;
using namespace cv;

/// <summary>
/// ͨ������Ǽ��������ڵ����ƥ��
/// </summary>
/// <param name="point_L"></param>
/// <param name="point_R"></param>
/// <param name="horizontal_L"></param>
/// <param name="vertical_L"></param>
/// <param name="horizontal_R"></param>
/// <param name="vertical_R"></param>
void line_search(Mat image_L, Mat image_R, Mat point_L, Mat point_R, Mat horizontal_L, Mat vertical_L, Mat horizontal_R, Mat vertical_R) {
    //1.ͨ������ͼ��ȡ������
    std::vector<cv::KeyPoint> joint_keypointsL;
    std::vector<cv::KeyPoint> joint_keypointsR;
    Ptr<ORB> orb = ORB::create(300, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    orb->detect(point_L, joint_keypointsL);
    orb->detect(point_R, joint_keypointsR);

    //2.ͨ��sift�㷨��ȡһЩǿ������
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    Ptr<cv::SiftFeatureDetector> sift = cv::SiftFeatureDetector::create();
    sift->detect(image_L, keypoints1);
    sift->detect(image_R, keypoints2);
    Mat descriptors1;
    Mat descriptors2;
    sift->compute(image_L, keypoints1, descriptors1);
    sift->compute(image_R, keypoints2, descriptors2);

    //ֱ�ӱ���ƥ��
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    /*matcher->match(descriptors1, descriptors2, matches);
    // ��ƥ��ͼ
    cv::Mat img_matches_bf;
    drawMatches(image_L, keypoints1, image_R, keypoints2, matches, img_matches_bf);
    //imshow("bf_matches", img_matches_bf);
    vector<DMatch> RR_matches = RANSAC_demo(image_L, image_R, keypoints1, keypoints2, matches);
*/
//KNN ƥ��

    std::vector<std::vector<DMatch> > knn_matches;
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    for (auto& knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }


    vector<DMatch> RR_matches = RANSAC_demo(image_L, image_R, keypoints1, keypoints2, good_matches);
    // ��ƥ��ͼ
    cv::Mat img_matches_knn;
    drawMatches(image_L, keypoints1, image_R, keypoints2, RR_matches, img_matches_knn, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("get strong keypoint from SIFT", img_matches_knn);

    //3.���ݵڶ���ǿ�����㣬��ȡ����ǿ������
    int number_primer_matches = RR_matches.size(); //��ʼǿ�����������
    weakFeature_mapping(RR_matches, keypoints1, keypoints2, joint_keypointsL, joint_keypointsR);


    //��RR_matchesɾ��ԭ��������
    for (int i = number_primer_matches; i < RR_matches.size(); i++) {
        RR_matches[i].queryIdx -= number_primer_matches;
        RR_matches[i].trainIdx -= number_primer_matches;
    }
    auto it1 = RR_matches.begin();
    RR_matches.erase(it1, it1 + number_primer_matches - 1);
    auto it2 = keypoints1.begin();
    keypoints1.erase(it2, it2 + number_primer_matches - 1);
    it2 = keypoints2.begin();
    keypoints2.erase(it2, it2 + number_primer_matches - 1);

    // ��ƥ��ͼ
    drawMatches(image_L, keypoints1, image_R, keypoints2, RR_matches, img_matches_knn, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("strong mesh keypoint", img_matches_knn);


    //4.��������ǿ�����㣬�������ڹǼܣ���y�����������������һ������
    //��������������������y�������̫������������Ͻ��ĵ㣬�����õ�ӽ���ͼ��ɾ����������������ֱ���ùǼ����ѱ��������
    verticalline_search(RR_matches, vertical_L, vertical_R, keypoints1, keypoints2, joint_keypointsL, joint_keypointsR);
    verticalline_search(RR_matches, vertical_L, vertical_R, keypoints1, keypoints2, joint_keypointsL, joint_keypointsR);
    drawMatches(image_L, keypoints1, image_R, keypoints2, RR_matches, img_matches_knn, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("verticalline_search", img_matches_knn);


    //5.�������ڹǼܣ���x�����������������һ������
    horizontalline_search(RR_matches, horizontal_L, horizontal_R, keypoints1, keypoints2, joint_keypointsL, joint_keypointsR);
    drawMatches(image_L, keypoints1, image_R, keypoints2, RR_matches, img_matches_knn, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("horizontalline_search", img_matches_knn);
    //test
    verticalline_search(RR_matches, vertical_L, vertical_R, keypoints1, keypoints2, joint_keypointsL, joint_keypointsR);
    //horizontalline_search(RR_matches, horizontal_L, horizontal_R, keypoints1, keypoints2, joint_keypointsL, joint_keypointsR);
    //verticalline_search(RR_matches, vertical_L, vertical_R, keypoints1, keypoints2, joint_keypointsL, joint_keypointsR);

    drawMatches(image_L, keypoints1, image_R, keypoints2, RR_matches, img_matches_knn, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("finall", img_matches_knn);
}
/// <summary>
/// �ظ��������ߣ�Ѱ��ƥ���������
/// </summary>
/// <param name="matches"></param>
/// <param name="vertical_L"></param>
/// <param name="vertical_R"></param>
/// <param name="keypoints1  ��֪ǿ������������"></param>
/// <param name="keypoints2"></param>
/// <param name="joint_keypointsL  �������񽻵�ͼ�õ���������"></param>
/// <param name="joint_keypointsR"></param>
void verticalline_search(vector<DMatch>& matches, Mat vertical_L, Mat vertical_R, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::KeyPoint> joint_keypointsL, std::vector<cv::KeyPoint> joint_keypointsR) {
    int size = matches.size();
    Mat vertical_L1;
    vertical_L.copyTo(vertical_L1);
    Mat vertical_R1;
    vertical_R.copyTo(vertical_R1);
    for (int i = 0; i < size; i++) {//����matches�е�ÿ����

        vector<float> length1;//�ظ����������ҵ������������ʼ��ľ���
        vector<float> length2;
        std::vector<cv::KeyPoint> find_keypoint1;//�ظ����������ҵ���������
        std::vector<cv::KeyPoint> find_keypoint2;
        //verticalline_search_func(keypoints1[i], vertical_L, length1, find_keypoint1, joint_keypointsL);
        verticalline_search_func(keypoints1[i], vertical_L1, length1, find_keypoint1, joint_keypointsL, keypoints2[i], vertical_R1, length2, find_keypoint2, joint_keypointsR);

        //����length1��length2����find_keypoint1��find_keypoint2�е����ݽ������
        printf("find_keypoint1.size() %d    find_keypoint2.size() %d\n", find_keypoint1.size(), find_keypoint2.size());
        int index1 = 0;
        int index2 = 0;
        for (index1 = 0; index1 < find_keypoint1.size(); ) {
            if (find_keypoint1.size() * find_keypoint2.size() == 0)
                break;
            if (index2 >= find_keypoint2.size())
                break;

            if (abs(length1[index1] - length2[index2]) <= 3) {
                // �������ӽ���������ƥ��

                //
            //ȱ�ٴ����ɾ�� 
            // 

                if (vector_find(keypoints1, find_keypoint1[index1]) < keypoints1.size() || vector_find(keypoints2, find_keypoint2[index2]) < keypoints2.size()) {
                    //���ظ���������
                    if (vector_find(keypoints1, find_keypoint1[index1]) == vector_find(keypoints2, find_keypoint2[index2])) {
                        index1++;
                        index2++;
                        continue;
                    }

                    else {  //��ʱ
                        index1++;
                        index2++;
                        continue;
                    };//ȱ�ٴ����ɾ��
                }

                DMatch newMatch;
                newMatch.queryIdx = keypoints1.size();
                newMatch.trainIdx = keypoints1.size();
                newMatch.distance = abs(length1[index1] - length2[index2]);
                keypoints1.push_back(find_keypoint1[index1]);
                keypoints2.push_back(find_keypoint2[index2]);
                matches.push_back(newMatch);

                index1++;
                index2++;

            }
            else if (length1[index1] > length2[index2] && length2[index2] > 0) {
                index2++;
            }
            else if (length1[index1] > length2[index2] && length2[index2] < 0) {
                index1++;
            }
            else if (length1[index1] < length2[index2] && length1[index1]>0) {
                index1++;
            }
            else if (length1[index1] < length2[index2] && length1[index1] < 0) {
                index2++;
            }
            if (index2 >= find_keypoint2.size())
                break;
        }
        printf("keypoints1.size() %d \n", keypoints1.size());

    }

    imshow("vertical_L1", vertical_L1);
}
/// <summary>
/// ����keypoint���ڹǼ��ߣ����ҵ��������㱣����find_keypoint�У�������ԭʼkeypoint�ľ��뱣����length��
/// </summary>
/// <param name="keypoint"></param>
/// <param name="vertical1"></param>
/// <param name="length"></param>
/// <param name="find_keypoint"></param>
/// <param name="joint_keypoints"></param>
void verticalline_search_func(KeyPoint keypoint, Mat& vertical, std::vector<float>& length, std::vector<cv::KeyPoint>& find_keypoint, std::vector<cv::KeyPoint> joint_keypoints, KeyPoint keypoint2, Mat& vertical2, std::vector<float>& length2, std::vector<cv::KeyPoint>& find_keypoint2, std::vector<cv::KeyPoint> joint_keypoints2) {
    int start_point_x = (int)keypoint.pt.x;
    int start_point_y = (int)keypoint.pt.y;
    int now_x = start_point_x;
    int now_y = start_point_y;
    //Mat vertical;
    //vertical1.copyTo(vertical);

    int start_point_x2 = (int)keypoint2.pt.x;
    int start_point_y2 = (int)keypoint2.pt.y;
    int now_x2 = start_point_x2;
    int now_y2 = start_point_y2;
    //Mat vertical2;
    //vertical12.copyTo(vertical2);

    //while (!is_endpoint(now_x, now_y, vertical) || !is_endpoint(now_x2, now_y2, vertical2)) {
    while (1) {
        //printf("now_x %d now_y %d\n", now_x, now_y);
        if (now_y >= vertical.cols - 2)
            break;
        vertical.at<uchar>(Point(now_x, now_y)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���y���������һ����
        if (now_y + 1 >= vertical.cols - 1)
            break;
        if (vertical.at<uchar>(Point(now_x + 1, now_y + 1)) != 0) {
            now_x++;
            now_y++;
        }
        else if (vertical.at<uchar>(Point(now_x - 1, now_y + 1)) != 0) {
            now_x--;
            now_y++;
        }
        else if (vertical.at<uchar>(Point(now_x, now_y + 1)) != 0) {
            now_y++;
        }
        else if (vertical.at<uchar>(Point(now_x + 1, now_y + 2)) != 0) {
            now_x++;
            now_y += 2;
        }
        else if (vertical.at<uchar>(Point(now_x - 1, now_y + 2)) != 0) {
            now_x--;
            now_y += 2;
        }
        else if (vertical.at<uchar>(Point(now_x, now_y + 2)) != 0) {
            now_y += 2;
        }
        else break;
        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints.size(); j++) {
            if (joint_keypoints[j].pt.x <= now_x + 1 && joint_keypoints[j].pt.x >= now_x - 1)
                if (joint_keypoints[j].pt.y < now_y + 1 && joint_keypoints[j].pt.y >= now_y) {
                    find_keypoint.push_back(joint_keypoints[j]);
                    length.push_back(joint_keypoints[j].pt.y - start_point_y);
                    break;
                }
        }


        if (now_y2 >= vertical2.cols - 2)
            break;
        vertical2.at<uchar>(Point(now_x2, now_y2)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���y���������һ����
        if (now_y2 + 1 >= vertical2.cols - 1)
            break;
        if (vertical2.at<uchar>(Point(now_x2 + 1, now_y2 + 1)) != 0) {
            now_x2++;
            now_y2++;
        }
        else if (vertical2.at<uchar>(Point(now_x2 - 1, now_y2 + 1)) != 0) {
            now_x2--;
            now_y2++;
        }
        else if (vertical2.at<uchar>(Point(now_x2, now_y2 + 1)) != 0) {
            now_y2++;
        }
        else if (vertical2.at<uchar>(Point(now_x2 + 1, now_y2 + 2)) != 0) {
            now_x2++;
            now_y2++;
        }
        else if (vertical2.at<uchar>(Point(now_x2 - 1, now_y2 + 2)) != 0) {
            now_x2--;
            now_y2++;
        }
        else if (vertical2.at<uchar>(Point(now_x2, now_y2 + 2)) != 0) {
            now_y2++;
        }
        else break;
        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints2.size(); j++) {
            if (joint_keypoints2[j].pt.x <= now_x2 + 1 && joint_keypoints2[j].pt.x >= now_x2 - 1)
                if (joint_keypoints2[j].pt.y < now_y2 + 1 && joint_keypoints2[j].pt.y >= now_y2) {
                    find_keypoint2.push_back(joint_keypoints2[j]);
                    length2.push_back(joint_keypoints2[j].pt.y - start_point_y2);
                    break;
                }
        }
    }

    now_x = start_point_x;
    now_y = start_point_y;
    now_x2 = start_point_x2;
    now_y2 = start_point_y2;


    //while (!is_endpoint(now_x, now_y, vertical) || !is_endpoint(now_x2, now_y2, vertical2)) {
    while (1) {
        //printf("now_x %d now_y %d\n", now_x, now_y);
        if (now_y <= 1)
            break;
        vertical.at<uchar>(Point(now_x, now_y)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���y���������һ����
        if (vertical.at<uchar>(Point(now_x + 1, now_y - 1)) != 0) {
            now_x++;
            now_y--;
        }
        else if (vertical.at<uchar>(Point(now_x - 1, now_y - 1)) != 0) {
            now_x--;
            now_y--;
        }
        else if (vertical.at<uchar>(Point(now_x, now_y - 1)) != 0) {
            now_y--;
        }
        else if (vertical.at<uchar>(Point(now_x + 1, now_y - 2)) != 0) {
            now_x++;
            now_y -= 2;
        }
        else if (vertical.at<uchar>(Point(now_x - 1, now_y - 2)) != 0) {
            now_x--;
            now_y -= 2;
        }
        else if (vertical.at<uchar>(Point(now_x, now_y - 2)) != 0) {
            now_y -= 2;
        }
        else break;

        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints.size(); j++) {
            if (joint_keypoints[j].pt.x <= now_x + 1 && joint_keypoints[j].pt.x >= now_x - 1)
                if (joint_keypoints[j].pt.y < now_y && joint_keypoints[j].pt.y >= now_y - 1) {
                    find_keypoint.push_back(joint_keypoints[j]);
                    length.push_back(joint_keypoints[j].pt.y - start_point_y);
                    break;
                }
        }

        if (now_y2 <= 1)
            break;
        vertical2.at<uchar>(Point(now_x2, now_y2)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���y���������һ����
        if (vertical2.at<uchar>(Point(now_x2 + 1, now_y2 - 1)) != 0) {
            now_x2++;
            now_y2--;
        }
        else if (vertical2.at<uchar>(Point(now_x2 - 1, now_y2 - 1)) != 0) {
            now_x2--;
            now_y2--;
        }
        else if (vertical2.at<uchar>(Point(now_x2, now_y2 - 1)) != 0) {
            now_y2--;
        }
        else if (vertical2.at<uchar>(Point(now_x2 + 1, now_y2 - 2)) != 0) {
            now_x2++;
            now_y2--;
        }
        else if (vertical2.at<uchar>(Point(now_x2 - 1, now_y2 - 2)) != 0) {
            now_x2--;
            now_y2--;
        }
        else if (vertical2.at<uchar>(Point(now_x2, now_y2 - 2)) != 0) {
            now_y2--;
        }
        else break;
        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints2.size(); j++) {
            if (joint_keypoints2[j].pt.x <= now_x2 + 1 && joint_keypoints2[j].pt.x >= now_x2 - 1)
                if (joint_keypoints2[j].pt.y < now_y2 && joint_keypoints2[j].pt.y >= now_y2 - 1) {
                    find_keypoint2.push_back(joint_keypoints2[j]);
                    length2.push_back(joint_keypoints2[j].pt.y - start_point_y2);
                    break;
                }
        }
    }
}
/// <summary>
/// �� x��y�Ƿ����߶εĶ˵�
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <param name="image"></param>
/// <returns></returns>
bool is_endpoint(int x, int y, Mat image) {
    if (x == image.rows - 1 || x == 0 || y == image.cols - 1 || y == 0)
        return true;
    if (image.at<uchar>(y, x + 1) != 0)
        return false;
    if (image.at<uchar>(y + 1, x + 1) != 0)
        return false;
    if (image.at<uchar>(y - 1, x + 1) != 0)
        return false;
    if (image.at<uchar>(y + 1, x) != 0)
        return false;
    if (image.at<uchar>(y - 1, x) != 0)
        return false;
    if (image.at<uchar>(y + 1, x - 1) != 0)
        return false;
    if (image.at<uchar>(y, x - 1) != 0)
        return false;
    if (image.at<uchar>(y - 1, x - 1) != 0)
        return false;
    return true;
}

/// <summary>
/// �ظ����ĺ��ߣ�Ѱ��ƥ���������
/// </summary>
/// <param name="matches"></param>
/// <param name="vertical_L"></param>
/// <param name="vertical_R"></param>
/// <param name="keypoints1  ��֪ǿ������������"></param>
/// <param name="keypoints2"></param>
/// <param name="joint_keypointsL  �������񽻵�ͼ�õ���������"></param>
/// <param name="joint_keypointsR"></param>
void horizontalline_search(vector<DMatch>& matches, Mat horizontal_L, Mat horizontal_R, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::KeyPoint> joint_keypointsL, std::vector<cv::KeyPoint> joint_keypointsR) {
    int size = matches.size();
    Mat horizontal_L1;
    horizontal_L.copyTo(horizontal_L1);
    Mat horizontal_R1;
    horizontal_R.copyTo(horizontal_R1);
    for (int i = 0; i < size; i++) {//����matches�е�ÿ����
        vector<float> length1;//�ظ����������ҵ������������ʼ��ľ���
        vector<float> length2;
        std::vector<cv::KeyPoint> find_keypoint1;//�ظ����������ҵ���������
        std::vector<cv::KeyPoint> find_keypoint2;
        //horizontalline_search_func(keypoints1[i], horizontal_L, length1, find_keypoint1, joint_keypointsL);
        horizontalline_search_func(keypoints1[i], horizontal_L1, length1, find_keypoint1, joint_keypointsL, keypoints2[i], horizontal_R1, length2, find_keypoint2, joint_keypointsR);

        //����length1��length2����find_keypoint1��find_keypoint2�е����ݽ������
        printf("find_keypoint1.size() %d    find_keypoint2.size() %d\n", find_keypoint1.size(), find_keypoint2.size());
        int index1 = 0;
        int index2 = 0;
        for (index1 = 0; index1 < find_keypoint1.size(); ) {
            if (find_keypoint1.size() * find_keypoint2.size() == 0)
                break;
            if (index2 >= find_keypoint2.size())
                break;
            if (abs(length1[index1] - length2[index2]) <= 3) {
                // �������ӽ���������ƥ��

                //
            //ȱ���ظ���ɾ���Լ������ɾ�� 
            // 

                if (vector_find(keypoints1, find_keypoint1[index1]) < keypoints1.size() || vector_find(keypoints2, find_keypoint2[index2]) < keypoints2.size()) {
                    //���ظ���������
                    if (vector_find(keypoints1, find_keypoint1[index1]) == vector_find(keypoints2, find_keypoint2[index2])) {
                        index1++;
                        index2++;
                        continue;
                    }
                    else;//ȱ�ٴ����ɾ��
                }

                DMatch newMatch;
                newMatch.queryIdx = keypoints1.size();
                newMatch.trainIdx = keypoints1.size();
                newMatch.distance = abs(length1[index1] - length2[index2]);
                keypoints1.push_back(find_keypoint1[index1]);
                keypoints2.push_back(find_keypoint2[index2]);
                matches.push_back(newMatch);

                index1++;
                index2++;
            }
            else if (length1[index1] > length2[index2] && length2[index2] > 0) {
                index2++;
            }
            else if (length1[index1] > length2[index2] && length2[index2] < 0) {
                index1++;
            }
            else if (length1[index1] < length2[index2] && length1[index1]>0) {
                index1++;
            }
            else if (length1[index1] < length2[index2] && length1[index1] < 0) {
                index2++;
            }
            if (index2 >= find_keypoint2.size())
                break;
        }
        printf("keypoints1.size() %d \n", keypoints1.size());

    }
    imshow("horizontal_L1", horizontal_L1);
}
/// <summary>
/// ����keypoint���ڹǼ��ߣ����ҵ��������㱣����find_keypoint�У�������ԭʼkeypoint�ľ��뱣����length��
/// </summary>
/// <param name="keypoint"></param>
/// <param name="vertical1"></param>
/// <param name="length"></param>
/// <param name="find_keypoint"></param>
/// <param name="joint_keypoints"></param>
void horizontalline_search_func(KeyPoint keypoint, Mat& horizontal, std::vector<float>& length, std::vector<cv::KeyPoint>& find_keypoint, std::vector<cv::KeyPoint> joint_keypoints, KeyPoint keypoint2, Mat& horizontal2, std::vector<float>& length2, std::vector<cv::KeyPoint>& find_keypoint2, std::vector<cv::KeyPoint> joint_keypoints2) {
    int start_point_x = (int)keypoint.pt.x;
    int start_point_y = (int)keypoint.pt.y;
    int now_x = start_point_x;
    int now_y = start_point_y;
    //Mat horizontal;
    //horizontal1.copyTo(horizontal);

    int start_point_x2 = (int)keypoint2.pt.x;
    int start_point_y2 = (int)keypoint2.pt.y;
    int now_x2 = start_point_x2;
    int now_y2 = start_point_y2;
    //Mat horizontal2;
    //horizontal12.copyTo(horizontal2);

    //while (!is_endpoint(now_x, now_y, horizontal) || !is_endpoint(now_x2, now_y2, horizontal2)) {
    while (1) {
        //printf("now_x %d now_y %d\n", now_x, now_y);
        if (now_x >= horizontal.rows - 2)
            break;
        horizontal.at<uchar>(Point(now_x, now_y)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���x���������һ����
        if (now_x + 1 >= horizontal.rows - 1)
            break;
        if (horizontal.at<uchar>(Point(now_x + 1, now_y + 1)) != 0) {
            now_x++;
            now_y++;
        }
        else if (horizontal.at<uchar>(Point(now_x + 1, now_y - 1)) != 0) {
            now_x++;
            now_y--;
        }
        else if (horizontal.at<uchar>(Point(now_x + 1, now_y)) != 0) {
            now_x++;
        }
        else break;
        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints.size(); j++) {
            if (joint_keypoints[j].pt.y <= now_y + 1 && joint_keypoints[j].pt.y >= now_y - 1)
                if (joint_keypoints[j].pt.x < now_x + 1 && joint_keypoints[j].pt.x >= now_x) {
                    find_keypoint.push_back(joint_keypoints[j]);
                    length.push_back(joint_keypoints[j].pt.x - start_point_x);
                    break;
                }
        }


        if (now_x2 >= horizontal2.rows - 2)
            break;
        horizontal2.at<uchar>(Point(now_x2, now_y2)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���x���������һ����
        if (now_x2 + 1 >= horizontal2.rows - 1)
            break;
        if (horizontal2.at<uchar>(Point(now_x2 + 1, now_y2 + 1)) != 0) {
            now_x2++;
            now_y2++;
        }
        else if (horizontal2.at<uchar>(Point(now_x2 + 1, now_y2 - 1)) != 0) {
            now_x2++;
            now_y2++;
        }
        else if (horizontal2.at<uchar>(Point(now_x2 + 1, now_y2)) != 0) {
            now_x2++;
        }
        else break;
        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints2.size(); j++) {
            if (joint_keypoints2[j].pt.y <= now_y2 + 1 && joint_keypoints2[j].pt.y >= now_y2 - 1)
                if (joint_keypoints2[j].pt.x < now_x2 + 1 && joint_keypoints2[j].pt.x >= now_x2) {
                    find_keypoint2.push_back(joint_keypoints2[j]);
                    length2.push_back(joint_keypoints2[j].pt.x - start_point_x2);
                    break;
                }
        }
    }

    now_x = start_point_x;
    now_y = start_point_y;
    now_x2 = start_point_x2;
    now_y2 = start_point_y2;


    //while (!is_endpoint(now_x, now_y, horizontal) || !is_endpoint(now_x2, now_y2, horizontal2)) {
    while (1) {
        //printf("now_x %d now_y %d\n", now_x, now_y);
        if (now_x <= 1)
            break;
        horizontal.at<uchar>(Point(now_x, now_y)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���x���������һ����
        if (horizontal.at<uchar>(Point(now_x - 1, now_y + 1)) != 0) {
            now_x--;
            now_y++;
        }
        else if (horizontal.at<uchar>(Point(now_x - 1, now_y - 1)) != 0) {
            now_x--;
            now_y--;
        }
        else if (horizontal.at<uchar>(Point(now_x - 1, now_y)) != 0) {
            now_x--;
        }
        else break;

        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints.size(); j++) {
            if (joint_keypoints[j].pt.y <= now_y + 1 && joint_keypoints[j].pt.y >= now_y - 1)
                if (joint_keypoints[j].pt.x < now_x && joint_keypoints[j].pt.x >= now_x - 1) {
                    find_keypoint.push_back(joint_keypoints[j]);
                    length.push_back(joint_keypoints[j].pt.x - start_point_x);
                    break;
                }
        }

        if (now_x2 <= 1)
            break;
        horizontal2.at<uchar>(Point(now_x2, now_y2)) = 0; //���Ѿ��������ĵ�����
        //Ѱ���߶���x���������һ����
        if (horizontal2.at<uchar>(Point(now_x2 - 1, now_y2 - 1)) != 0) {
            now_x2--;
            now_y2--;
        }
        else if (horizontal2.at<uchar>(Point(now_x2 - 1, now_y2 + 1)) != 0) {
            now_x2--;
            now_y2++;
        }
        else if (horizontal2.at<uchar>(Point(now_x2 - 1, now_y2)) != 0) {
            now_x2--;
        }
        else break;
        //���ҵ�ǰ���긽����������������
        for (int j = 0; j < joint_keypoints2.size(); j++) {
            if (joint_keypoints2[j].pt.y <= now_y2 + 1 && joint_keypoints2[j].pt.y >= now_y2 - 1)
                if (joint_keypoints2[j].pt.x < now_x2 && joint_keypoints2[j].pt.x >= now_x2 - 1) {
                    find_keypoint2.push_back(joint_keypoints2[j]);
                    length2.push_back(joint_keypoints2[j].pt.x - start_point_x2);
                    break;
                }
        }
    }
}

