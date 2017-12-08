#pragma once

#include "opencv2/opencv.hpp"
using namespace cv;

class ImageTransformData
{
public:
	ImageTransformData()
	{
		params.numsample = 1000;
		params.maxintensitydev = 40;
		params.maxxangle = 0.15; // PI/20
		params.maxyangle = 1.0; // PI/3
		params.maxzangle = 0.15; // PI/20
		params.minrad = 2;
		params.maxrad = 6;
		params.maxincl = 0.25;
		params.winsize = 24;
		params.maxscale = -1.0;
		params.random = false;
	};

	Mat img_src, img_tran, mask;

	std::string bgname;
	double maxscale, scale;
	int width, height;
	double bg_fill_color;
	double transparent_color_low, transparent_color_high;
	int x, y;
	double r;
	double i;
	double xangle, yangle, zangle;
	double light_x, light_y, light_z;
	double light_color, light_intensity;
	double object_albedo;
	double object_Ka, object_Kd, object_Ks;
	double object_shininess;

	struct
	{
		int numsample;
		std::string bg_fill_color;
		int transparent_color_low, transparent_color_high;
		bool noise_removal;
		int maxintensitydev;
		double maxxangle;
		double maxyangle;
		double maxzangle;
		double minrad;
		double maxrad;
		double maxincl;
		int winsize;
		double maxscale;
		bool random;
		double albedo_min, albedo_max;
		double ambient;
		double specular_min, specular_max;
		double shininess_min, shininess_max;
		double light_color;
		double light_intensity_min, light_intensity_max;
		double light_dir_dev_max;
	} params;
};
