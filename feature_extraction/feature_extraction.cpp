#include "feature_extraction.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;
/// <summary>
/// 闭操作
/// </summary>
/// <param name="src=源图像"></param>
/// <param name="method=kernel类型，0为水平结构，1为垂直结构，3为矩形结构"></param>
/// <param name="ksize=kernel大小"></param>
/// <returns></returns>
Mat close_operation(Mat src,int method=2,int ksize=3) {   // 闭操作
    Mat  dst;
    //水平结构元素
    //Mat hline = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1),
        //Point(-1, -1));
    Mat hline = getStructuringElement(MORPH_RECT, Size(ksize, 1),
        Point(-1, -1));
    //垂直结构元素
    //Mat vline = getStructuringElement(MORPH_RECT, Size(1, src.rows / 16),
       // Point(-1, -1));
    Mat vline = getStructuringElement(MORPH_RECT, Size(1, ksize),
        Point(-1, -1));
      
    // 矩形结构
    Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize),
        Point(-1, -1));

    switch (method) {
    case 0:kernel = hline; break;
    case 1:kernel = vline; break;
    case 2:kernel=kernel; break;
    }
    Mat temp;
    dilate(src, temp, kernel);
    erode(temp, dst, kernel);

    //imshow("闭操作后结果", dst);
    return dst;
}
/// <summary>
/// 
/// </summary>
/// <param name="src"></param>
/// <returns></returns>
Mat get_joints(Mat src, int scale_H, int scale_V) {
    Mat binImg;   //二值化
    GaussianBlur(src, src, Size(5, 5), 1);   //高斯滤波
    adaptiveThreshold(src, binImg, 255,
        ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);  //原始值-2

    
    binImg= close_operation(binImg);//闭操作
    //imshow("binary image", binImg);


    //使用二值化后的图像来获取表格横纵的线
    Mat horizontal = binImg.clone();
    Mat vertical = binImg.clone();

    //int scale_H = 140; //这个值越大，检测到的直线越多
    //int scale_V = 60; //这个值越大，检测到的直线越多
    //int scale_H = 120; //这个值越大，检测到的横线越多
    //int scale_V = 100; //这个值越大，检测到的竖线越多


    int horizontalsize = horizontal.cols / scale_H;
    // 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
    // 先腐蚀再膨胀
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    horizontal = close_operation(horizontal,0,5);//闭操作
    horizontal = close_operation(horizontal, 0, 7);//闭操作
    //imshow("horizontal", horizontal);

    int verticalsize = vertical.rows / scale_V;
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));
    vertical = close_operation(vertical, 1, 5);//闭操作
    vertical = close_operation(vertical, 1, 7);//闭操作
    //imshow("vertical", vertical);

   
    /*Mat dst1;                        //利用掩膜操作对原始图像光条进行提取
    dst1 = mask(src, horizontal);
    imshow("horizontal", dst1);
    Mat dst2;
    //src.copyTo(dst2, vertical);
    dst2 = mask(src, vertical);
    imshow("vertical", dst2);
*/
    //图像细化，骨骼化  
    cv::Mat dst1;
    cvHilditchThin1(horizontal, dst1);
    //imshow("dst1", dst1);
    cv::Mat dst2;
    cvHilditchThin1(vertical, dst2);
    //imshow("dst2", dst2);

    

    Mat mask = horizontal + vertical;
    mask = dst1 + dst2;
    mask = mask + src;
    //imshow("src", src);

    //imshow("mask", mask);

    

    Mat joints;   //获取交点
    bitwise_and(horizontal, vertical, joints);
    bitwise_and(dst1, dst2, joints);

    joints = close_operation(joints);//闭操作
    //imshow("joints", joints);

    joints = preprocess(src, joints, 11, 130);


    return joints;
}

/// <summary>
/// OpenCV 对比度受限的自适应直方图均衡化(CLAHE)算法，包括以下三个
/// </summary>
/// <param name="input"></param>
/// <param name="chls"></param>
static void color_transfer_with_spilt(cv::Mat& input, std::vector<cv::Mat>& chls)
{
    cv::cvtColor(input, input, cv::COLOR_BGR2YCrCb);
    cv::split(input, chls);
}

static void color_retransfer_with_merge(cv::Mat& output, std::vector<cv::Mat>& chls)
{
    cv::merge(chls, output);
    cv::cvtColor(output, output, cv::COLOR_YCrCb2BGR);
}

cv::Mat clahe_deal(cv::Mat& src)
{
    cv::Mat ycrcb = src.clone();
    std::vector<cv::Mat> channels;

    color_transfer_with_spilt(ycrcb, channels);

    cv::Mat clahe_img;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    // 直方图的柱子高度大于计算后的ClipLimit的部分被裁剪掉，然后将其平均分配给整张直方图
    // 从而提升整个图像
    clahe->setClipLimit(4.);    // (int)(4.*(8*8)/256)
    clahe->setTilesGridSize(Size(8, 8)); // 将图像分为8*8块
    clahe->apply(channels[0], clahe_img);
    channels[0].release();
    clahe_img.copyTo(channels[0]);
    color_retransfer_with_merge(ycrcb, channels);
    return ycrcb;
}
/// <summary>
/// 掩膜操作，利用maskimage对src抠图
/// </summary>
/// <param name="src 原始图像"></param>
/// <param name="maskimage 掩膜"></param>
/// <returns></returns>
Mat mask(Mat src,Mat maskimage) {
    int rowNumber = src.rows;
    int colNumber = src.cols;
    Mat image = src.clone();
    for (int i = 0; i < rowNumber; i++)
    {
        for (int j = 0; j < colNumber; j++)
        {   
            if (maskimage.at<uchar>(i, j) == 0)
                image.at<uchar>(i, j) = 0;
        }
    }
    return image;
}

/**
* @brief 对输入图像进行细化,骨骼化
* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param dst为输出图像
* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
void cvHilditchThin1(cv::Mat& src, cv::Mat& dst)
{
    //http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
    //算法有问题，得不到想要的效果
    if (src.type() != CV_8UC1)
    {
        printf("只能处理二值或灰度图像\n");
        return;
    }
    //非原地操作时候，copy src到dst
    if (dst.data != src.data)
    {
        src.copyTo(dst);
    }

    int i, j;
    int width, height;
    //之所以减2，是方便处理8邻域，防止越界
    width = src.cols - 2;
    height = src.rows - 2;
    int step = src.step;
    int  p2, p3, p4, p5, p6, p7, p8, p9;
    uchar* img;
    bool ifEnd;
    int A1;
    cv::Mat tmpimg;
    while (1)
    {
        dst.copyTo(tmpimg);
        ifEnd = false;
        img = tmpimg.data + step;
        for (i = 2; i < height; i++)
        {
            img += step;
            for (j = 2; j < width; j++)
            {
                uchar* p = img + j;
                A1 = 0;
                if (p[0] > 0)
                {
                    if (p[-step] == 0 && p[-step + 1] > 0) //p2,p3 01模式
                    {
                        A1++;
                    }
                    if (p[-step + 1] == 0 && p[1] > 0) //p3,p4 01模式
                    {
                        A1++;
                    }
                    if (p[1] == 0 && p[step + 1] > 0) //p4,p5 01模式
                    {
                        A1++;
                    }
                    if (p[step + 1] == 0 && p[step] > 0) //p5,p6 01模式
                    {
                        A1++;
                    }
                    if (p[step] == 0 && p[step - 1] > 0) //p6,p7 01模式
                    {
                        A1++;
                    }
                    if (p[step - 1] == 0 && p[-1] > 0) //p7,p8 01模式
                    {
                        A1++;
                    }
                    if (p[-1] == 0 && p[-step - 1] > 0) //p8,p9 01模式
                    {
                        A1++;
                    }
                    if (p[-step - 1] == 0 && p[-step] > 0) //p9,p2 01模式
                    {
                        A1++;
                    }
                    p2 = p[-step] > 0 ? 1 : 0;
                    p3 = p[-step + 1] > 0 ? 1 : 0;
                    p4 = p[1] > 0 ? 1 : 0;
                    p5 = p[step + 1] > 0 ? 1 : 0;
                    p6 = p[step] > 0 ? 1 : 0;
                    p7 = p[step - 1] > 0 ? 1 : 0;
                    p8 = p[-1] > 0 ? 1 : 0;
                    p9 = p[-step - 1] > 0 ? 1 : 0;
                    //计算AP2,AP4
                    int A2, A4;
                    A2 = 0;
                    //if(p[-step]>0)
                    {
                        if (p[-2 * step] == 0 && p[-2 * step + 1] > 0) A2++;
                        if (p[-2 * step + 1] == 0 && p[-step + 1] > 0) A2++;
                        if (p[-step + 1] == 0 && p[1] > 0) A2++;
                        if (p[1] == 0 && p[0] > 0) A2++;
                        if (p[0] == 0 && p[-1] > 0) A2++;
                        if (p[-1] == 0 && p[-step - 1] > 0) A2++;
                        if (p[-step - 1] == 0 && p[-2 * step - 1] > 0) A2++;
                        if (p[-2 * step - 1] == 0 && p[-2 * step] > 0) A2++;
                    }


                    A4 = 0;
                    //if(p[1]>0)
                    {
                        if (p[-step + 1] == 0 && p[-step + 2] > 0) A4++;
                        if (p[-step + 2] == 0 && p[2] > 0) A4++;
                        if (p[2] == 0 && p[step + 2] > 0) A4++;
                        if (p[step + 2] == 0 && p[step + 1] > 0) A4++;
                        if (p[step + 1] == 0 && p[step] > 0) A4++;
                        if (p[step] == 0 && p[0] > 0) A4++;
                        if (p[0] == 0 && p[-step] > 0) A4++;
                        if (p[-step] == 0 && p[-step + 1] > 0) A4++;
                    }


                    //printf("p2=%d p3=%d p4=%d p5=%d p6=%d p7=%d p8=%d p9=%d\n", p2, p3, p4, p5, p6,p7, p8, p9);
                    //printf("A1=%d A2=%d A4=%d\n", A1, A2, A4);
                    if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7 && A1 == 1)
                    {
                        if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
                        {
                            dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
                            ifEnd = true;
                            //printf("\n");

                            //PrintMat(dst);
                        }
                    }
                }
            }
        }
        //printf("\n");
        //PrintMat(dst);
        //PrintMat(dst);
        //已经没有可以细化的像素了，则退出迭代
        if (!ifEnd) break;
    }
}
/// <summary>
/// 去除中心光斑
/// </summary>
/// <param name="src 原图像"></param>
/// <param name="kernal 大小为Ksize*Ksize"></param>
/// <param name="H 阈值 "></param>
/// <returns></returns>
Mat preprocess(Mat src1, Mat src2, int Ksize, int H) {
    int rowNumber = src1.rows;
    int colNumber = src1.cols;
    Mat image = src1.clone();
    Mat image2 = src2.clone();
    for (int i = Ksize / 2; i < rowNumber - Ksize / 2; i++)
    {
        for (int j = Ksize / 2; j < colNumber - Ksize / 2; j++)
        {
            int mean = 0;
            for (int a = -Ksize / 2; a < Ksize / 2; a++)
                for (int b = -Ksize / 2; b < Ksize / 2; b++)
                    mean += image.at<uchar>(i + a, j + b) / (Ksize * Ksize);
            if (mean > H) {
                for (int a = -Ksize / 2; a < Ksize / 2; a++)
                    for (int b = -Ksize / 2; b < Ksize / 2; b++)
                        image2.at<uchar>(i + a, j + b) = 0;
            }

        }
    }
    return image2;
}
