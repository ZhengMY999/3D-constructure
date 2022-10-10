
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp> 
#include <iostream>
#include <opencv2\imgproc\types_c.h>
#include "feature_extraction.h"
#include "feature_mapping.h"

using namespace cv;
using namespace std;

//�ɵ����� ˫���˲���������ֵ��������func2�и�ʴģ���С��clahe�㷨�еĲ���



int main(int argc, char* argv[])
{
    Mat image_L = imread("C:/Users/DELL/Desktop/�㲿�ṹ��/��Ƭ/10.1/side_50x50_right.jpg", IMREAD_GRAYSCALE);
    Mat image_R = imread("C:/Users/DELL/Desktop/�㲿�ṹ��/��Ƭ/10.1/side_50x50_left.jpg", IMREAD_GRAYSCALE);
    //imshow("image_R", image_R);
    //resize(image, image, Size(image.cols / 4, image.rows / 4), 0, 0, INTER_NEAREST); //��СͼƬ  
    //image_R = preprocess(image_R, 11, 130);
    //imshow("preprocess", image_R);

    //����У��
    Undistort(image_L, image_R, image_L, image_R);
    
    //��ȡ����
    Mat mat1 = get_joints(image_R, 140, 110);
    Mat mat2 = get_joints(image_L, 140, 110);
    imshow("mat1", mat1);
    imshow("mat2", mat2);

    //�������ƥ��
    sift(image_R, image_L, mat1, mat2);

    cv::waitKey(0);
    waitKey();
    return 0;
}

