#include "feature_extraction.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;
/// <summary>
/// �ղ���
/// </summary>
/// <param name="src=Դͼ��"></param>
/// <param name="method=kernel���ͣ�0Ϊˮƽ�ṹ��1Ϊ��ֱ�ṹ��3Ϊ���νṹ"></param>
/// <param name="ksize=kernel��С"></param>
/// <returns></returns>
Mat close_operation(Mat src,int method=2,int ksize=3) {   // �ղ���
    Mat  dst;
    //ˮƽ�ṹԪ��
    //Mat hline = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1),
        //Point(-1, -1));
    Mat hline = getStructuringElement(MORPH_RECT, Size(ksize, 1),
        Point(-1, -1));
    //��ֱ�ṹԪ��
    //Mat vline = getStructuringElement(MORPH_RECT, Size(1, src.rows / 16),
       // Point(-1, -1));
    Mat vline = getStructuringElement(MORPH_RECT, Size(1, ksize),
        Point(-1, -1));
      
    // ���νṹ
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

    //imshow("�ղ�������", dst);
    return dst;
}
/// <summary>
/// 
/// </summary>
/// <param name="src"></param>
/// <returns></returns>
Mat get_joints(Mat src, int scale_H, int scale_V) {
    Mat binImg;   //��ֵ��
    GaussianBlur(src, src, Size(5, 5), 1);   //��˹�˲�
    adaptiveThreshold(src, binImg, 255,
        ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);  //ԭʼֵ-2

    
    binImg= close_operation(binImg);//�ղ���
    //imshow("binary image", binImg);


    //ʹ�ö�ֵ�����ͼ������ȡ�����ݵ���
    Mat horizontal = binImg.clone();
    Mat vertical = binImg.clone();

    //int scale_H = 140; //���ֵԽ�󣬼�⵽��ֱ��Խ��
    //int scale_V = 60; //���ֵԽ�󣬼�⵽��ֱ��Խ��
    //int scale_H = 120; //���ֵԽ�󣬼�⵽�ĺ���Խ��
    //int scale_V = 100; //���ֵԽ�󣬼�⵽������Խ��


    int horizontalsize = horizontal.cols / scale_H;
    // Ϊ�˻�ȡ����ı���ߣ����ø�ʴ�����͵Ĳ�������Ϊһ���Ƚϴ�ĺ���ֱ��
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
    // �ȸ�ʴ������
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    horizontal = close_operation(horizontal,0,5);//�ղ���
    horizontal = close_operation(horizontal, 0, 7);//�ղ���
    //imshow("horizontal", horizontal);

    int verticalsize = vertical.rows / scale_V;
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));
    vertical = close_operation(vertical, 1, 5);//�ղ���
    vertical = close_operation(vertical, 1, 7);//�ղ���
    //imshow("vertical", vertical);

   
    /*Mat dst1;                        //������Ĥ������ԭʼͼ�����������ȡ
    dst1 = mask(src, horizontal);
    imshow("horizontal", dst1);
    Mat dst2;
    //src.copyTo(dst2, vertical);
    dst2 = mask(src, vertical);
    imshow("vertical", dst2);
*/
    //ͼ��ϸ����������  
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

    

    Mat joints;   //��ȡ����
    bitwise_and(horizontal, vertical, joints);
    bitwise_and(dst1, dst2, joints);

    joints = close_operation(joints);//�ղ���
    //imshow("joints", joints);

    joints = preprocess(src, joints, 11, 130);


    return joints;
}

/// <summary>
/// OpenCV �Աȶ����޵�����Ӧֱ��ͼ���⻯(CLAHE)�㷨��������������
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
    // ֱ��ͼ�����Ӹ߶ȴ��ڼ�����ClipLimit�Ĳ��ֱ��ü�����Ȼ����ƽ�����������ֱ��ͼ
    // �Ӷ���������ͼ��
    clahe->setClipLimit(4.);    // (int)(4.*(8*8)/256)
    clahe->setTilesGridSize(Size(8, 8)); // ��ͼ���Ϊ8*8��
    clahe->apply(channels[0], clahe_img);
    channels[0].release();
    clahe_img.copyTo(channels[0]);
    color_retransfer_with_merge(ycrcb, channels);
    return ycrcb;
}
/// <summary>
/// ��Ĥ����������maskimage��src��ͼ
/// </summary>
/// <param name="src ԭʼͼ��"></param>
/// <param name="maskimage ��Ĥ"></param>
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
* @brief ������ͼ�����ϸ��,������
* @param srcΪ����ͼ��,��cvThreshold�����������8λ�Ҷ�ͼ���ʽ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
* @param dstΪ���ͼ��
* @return Ϊ��srcϸ��������ͼ��,��ʽ��src��ʽ��ͬ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
*/
void cvHilditchThin1(cv::Mat& src, cv::Mat& dst)
{
    //http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
    //�㷨�����⣬�ò�����Ҫ��Ч��
    if (src.type() != CV_8UC1)
    {
        printf("ֻ�ܴ����ֵ��Ҷ�ͼ��\n");
        return;
    }
    //��ԭ�ز���ʱ��copy src��dst
    if (dst.data != src.data)
    {
        src.copyTo(dst);
    }

    int i, j;
    int width, height;
    //֮���Լ�2���Ƿ��㴦��8���򣬷�ֹԽ��
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
                    if (p[-step] == 0 && p[-step + 1] > 0) //p2,p3 01ģʽ
                    {
                        A1++;
                    }
                    if (p[-step + 1] == 0 && p[1] > 0) //p3,p4 01ģʽ
                    {
                        A1++;
                    }
                    if (p[1] == 0 && p[step + 1] > 0) //p4,p5 01ģʽ
                    {
                        A1++;
                    }
                    if (p[step + 1] == 0 && p[step] > 0) //p5,p6 01ģʽ
                    {
                        A1++;
                    }
                    if (p[step] == 0 && p[step - 1] > 0) //p6,p7 01ģʽ
                    {
                        A1++;
                    }
                    if (p[step - 1] == 0 && p[-1] > 0) //p7,p8 01ģʽ
                    {
                        A1++;
                    }
                    if (p[-1] == 0 && p[-step - 1] > 0) //p8,p9 01ģʽ
                    {
                        A1++;
                    }
                    if (p[-step - 1] == 0 && p[-step] > 0) //p9,p2 01ģʽ
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
                    //����AP2,AP4
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
                            dst.at<uchar>(i, j) = 0; //����ɾ�����������õ�ǰ����Ϊ0
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
        //�Ѿ�û�п���ϸ���������ˣ����˳�����
        if (!ifEnd) break;
    }
}
/// <summary>
/// ȥ�����Ĺ��
/// </summary>
/// <param name="src ԭͼ��"></param>
/// <param name="kernal ��СΪKsize*Ksize"></param>
/// <param name="H ��ֵ "></param>
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
