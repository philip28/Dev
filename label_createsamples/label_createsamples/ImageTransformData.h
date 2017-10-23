#pragma once

#include "opencv2/opencv.hpp"
using namespace cv;

class ImageTransformData
{
public:
	ImageTransformData();

	Mat src;
	Mat mask;
	Mat trans_img;
	Mat trans_mask;

	float scale;
	int width, height;
	int x, y;
	double r1, r2;
	double phi;
	double hangle, vangle;
	int tr_width, tr_height;

	struct
	{
		unsigned numsample;
		unsigned bgcolor;
		unsigned bgthreshold;
		unsigned maxintensitydev;
		double maxhangle;
		double maxvangle;
		double minrad;
		double maxrad;
		double maxrot;
		unsigned winwidth;
		unsigned winheight;
		double maxscale;
		bool random;
	} params;
};
