#include "stdafx.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <math.h>
#include <iostream>
using namespace std;
#define PI 3.14159265358979323846

float VectorVote(int _nbins, float angle) {
	float step = 180.0 / _nbins;
	for (float i = 1.0; i <= _nbins; i++) {
		if (angle >= (i - 1)*step-90&&angle <= i*step-90)
			return i;
	}
}

float *HoG(cv::Mat &src,cv::Size _winSize,cv::Size _blockSize,cv::Size _blockStride,cv::Size _cellSize,int _nbins) {
	//伽马颜色规范化
	cv::Mat inputimage;
	cv::resize(src, inputimage, _winSize);
	cv::Mat image = cv::Mat(inputimage.rows, inputimage.cols, CV_32FC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			image.at<cv::Vec3f>(i, j)[0] = sqrt(inputimage.at<cv::Vec3b>(i, j)[0]);
			image.at<cv::Vec3f>(i, j)[1] = sqrt(inputimage.at<cv::Vec3b>(i, j)[1]);
			image.at<cv::Vec3f>(i, j)[2] = sqrt(inputimage.at<cv::Vec3b>(i, j)[2]);
		}
	}
	//cv::namedWindow("Gamma");
	//cv::imshow("Gamma",image);
	//梯度计算
	float k13[] = { -1,0,1 };
	float k31[3][1] = { -1,0,1 };
	cv::Mat Kore = cv::Mat(1, 3, CV_32FC1, k13);
	cv::Mat Kore1 = cv::Mat(3, 1, CV_32FC1, k31);
	cv::Mat dstx, dsty;
	cv::filter2D(image, dstx, image.depth(), Kore);
	cv::filter2D(image, dsty, image.depth(), Kore1);
	cv::Mat gradientx = cv::Mat(dstx.rows, dstx.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat gradienty = cv::Mat(dsty.rows, dsty.cols, CV_32FC1, cv::Scalar(0));
	//以范数最大者作为该点的梯度向量
	for (int i = 0; i < dstx.rows; i++) {
		for (int j = 0; j < dstx.cols; j++) {
			float ax = dstx.at<cv::Vec3f>(i, j)[0];
			float bx = dstx.at<cv::Vec3f>(i, j)[1];
			float cx = dstx.at<cv::Vec3f>(i, j)[2];
			float ay = dsty.at<cv::Vec3f>(i, j)[0];
			float by = dsty.at<cv::Vec3f>(i, j)[1];
			float cy = dsty.at<cv::Vec3f>(i, j)[2];
			if (ax*ax + ay*ay > bx*bx + by*by) {
				if (ax*ax + ay*ay > cx*cx + cy*cy) {
					gradientx.at<float>(i, j) = ax;
					gradienty.at<float>(i, j) = ay;
				}
				else {
					gradientx.at<float>(i, j) = cx;
					gradienty.at<float>(i, j) = cy;
				}
			}
			else
			{
				if (bx*bx + by*by > cx*cx + cy*cy) {
					gradientx.at<float>(i, j) = bx;
					gradienty.at<float>(i, j) = by;
				}
				else {
					gradientx.at<float>(i, j) = cx;
					gradienty.at<float>(i, j) = cy;
				}
			}
		}
	}
	//cv::namedWindow("gradientx");
	//cv::imshow("gradientx",gradientx);
	//cv::namedWindow("gradienty");
	//cv::imshow("gradienty",gradienty);
	cv::Mat vote = cv::Mat(image.rows, image.cols, CV_32FC2, cv::Scalar(0));
	for (int i = 0; i < vote.rows; i++)
		for (int j = 0; j < vote.cols; j++) {
			float y = gradienty.at<float>(i, j);
			float x = gradientx.at<float>(i, j);
			if (x != 0) {
				vote.at<cv::Vec2f>(i, j)[0] = VectorVote(_nbins, atan(y / x) / PI*180.0);
				vote.at<cv::Vec2f>(i, j)[1] = sqrt(y*y + x*x);
			}
			else {
				if (y >= 0) {
					vote.at<cv::Vec2f>(i, j)[0] = VectorVote(_nbins, 90);
					vote.at<cv::Vec2f>(i, j)[1] = sqrt(y*y + x*x);
				}
				else {
					vote.at<cv::Vec2f>(i, j)[0] = VectorVote(_nbins, -90);
					vote.at<cv::Vec2f>(i, j)[1] = sqrt(y*y + x*x);
				}
			}
		}
	//计算梯度方向直方图，对cell内每个像素用梯度方向在直方图中进行加权投影
	float ***cell;
	cell = (float***)malloc(sizeof(float**)*(vote.rows / _cellSize.width));
	for (int i = 0; i < vote.rows / _cellSize.width; i++)
		cell[i] = (float**)malloc(sizeof(float*)*(vote.cols / _cellSize.height));
	for(int i=0;i<vote.rows/_cellSize.width;i++)
		for (int j = 0; j < vote.cols / _cellSize.height; j++) {
			cell[i][j] = (float*)malloc(sizeof(float)*_nbins);
			for (int k = 0; k < _nbins; k++)
				cell[i][j][k] = 0;
			int startwidth = i*_cellSize.width;
			int startheight = i*_cellSize.height;
			for (int w = startwidth; w < startwidth + _cellSize.width; w++) {
				for (int h = startheight; h < startheight + _cellSize.height; h++) {
					int position = (int)vote.at<cv::Vec2f>(w, h)[0];
					cell[i][j][position] += vote.at<cv::Vec2f>(w, h)[1];
				}
			}
		}
	//计算block的个数
	int x_step_num = (_winSize.width - _blockSize.width) / _blockStride.width + 1;
	int y_step_num = (_winSize.height - _blockSize.height) / _blockStride.height + 1;
	//HoG特征
	float *HoGFeature = (float*)malloc(sizeof(float)*(_blockSize.width / _cellSize.width)*(_blockSize.height / _cellSize.height)*_nbins*x_step_num*y_step_num);
	int HoGCount = 0;
	//初始化Hog特征向量
	for (int i = 0; i < y_step_num; i++) {
		for (int j = 0; j < x_step_num; j++) {
			//求当前的block的HoG直方图
			int startCellx = (i*_blockStride.width) / _cellSize.width;
			int startCelly = (j*_blockStride.height) / _cellSize.height;
			float *blockFeature = (float*)malloc(sizeof(float)*(_blockSize.width / _cellSize.width)*(_blockSize.height / _cellSize.height)*_nbins);
			//将block内所有cell的特征向量串联起来得到该block的HOG特征
			int count = 0;
			int sumSquares = 0;
			for (int bx = 0; bx < _blockSize.width / _cellSize.width; bx++) {
				for (int by = 0; by < _blockSize.height / _cellSize.height; by++) {
					for (int bNum = 0; bNum < _nbins; bNum++) {
						blockFeature[count] = cell[startCellx+bx][startCelly+by][bNum];
						sumSquares += blockFeature[count] * blockFeature[count];
						count++;
					}
				}
			}
			//归一化 L2-norm
			for (int b = 0; b < (_blockSize.width / _cellSize.width)*(_blockSize.height / _cellSize.height)*_nbins; b++) {
				blockFeature[b] = blockFeature[b] / sqrt(sumSquares);
				HoGFeature[HoGCount] = blockFeature[b];
				HoGCount++;
			}
			free(blockFeature);
		}
	}
	//HoG特征计算完毕
	//开始free
	free(cell);
	return HoGFeature;
}


int main() {
	cv::Mat image = cv::imread("E://Image//1.jpg");
	if (image.empty()) {
		cout << "该图片无效" << endl;
		return 0;
	}
	cout << "得到HoG特征向量" << endl;
	float *HoGFeature=HoG(image,cv::Size(128,64),cv::Size(16,16),cv::Size(8,8),cv::Size(8,8),9);
	for (int i = 0; i < (16/8) * (16/8) * ((128 - 8) / 8) * ((64 - 8) / 8)*9; i++)
		cout << HoGFeature[i] << endl;

	cvWaitKey();
	return 0;
}
