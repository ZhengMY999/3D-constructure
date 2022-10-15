
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp> 
#include <iostream>
#include <opencv2/imgproc/types_c.h>
#include "feature_extraction.h"
#include "feature_mapping.h"
#include "thinImage.h"
using namespace cv;
using namespace std;

//可调参数 双边滤波参数，二值化参数，func2中腐蚀模板大小，clahe算法中的参数



int main(int argc, char* argv[])
{
    Mat image_L = imread("/home/wangzha/Desktop/3D-constructure/resources/side_50x50_left.jpg", IMREAD_GRAYSCALE);
    Mat image_R = imread("/home/wangzha/Desktop/3D-constructure/resources/side_50x50_right.jpg", IMREAD_GRAYSCALE);


    //极线校正
    Undistort(image_L, image_R, image_L, image_R);

    Rect react1(1896, 845, 694, 587);
    Rect react2(1236, 854, 694, 587);

    image_L = image_L(react1);
    image_R = image_R(react2);



    //获取交点
    Mat mat1 = get_joints(image_R, 8, 8);
    Mat mat2 = get_joints(image_L, 8, 8);
    imshow("mat1", mat1);
    imshow("mat2", mat2);

    //交点进行匹配
    //surf(image_R, image_L, mat1, mat2);
     
    //10.15 添加
    //获取交点
    Mat pointR = get_joints(image_R, 8, 8);
    Mat pointL = get_joints(image_L, 8, 8);
    //imshow("point1", point1);
    //imshow("point2", point2);


    //获取横竖骨架
    Mat horizontal_L;
    Mat vertical_L;
    Mat horizontal_R;
    Mat vertical_R;
    get_skeleton(image_L, 5, 5, horizontal_L, vertical_L);
    get_skeleton(image_R, 5, 5, horizontal_R, vertical_R);
    line_search(image_L, image_R, pointL, pointR, horizontal_L, vertical_L, horizontal_R, vertical_R);
    
    cv::waitKey(0);
    return 0;
}

