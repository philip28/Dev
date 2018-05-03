#pragma once

#include "Plane.h"
#include "Cylinder.h"
#include "ImageReader.h"
#include "InfoWriter.h"
#include "shading.h"
#include "opencv2/opencv.hpp"
#include "RandGen.h"

typedef struct
{
	std::string image_dir;
	std::string image_file;
	std::string background_file;
	std::string background_dir;
	std::string background_list;
	std::string output_dir;
	std::string output_dir_bg;
	std::string info_file;
	int num_samples = 1000;
	int num_patches = 1;
	std::string bg_color = "transparent";
	std::string fill_color = "transparent";
	std::string output_type = "embed";
	bool randomize = false;
	bool grayscale = false;
	int transparent_color_low = 0, transparent_color_high = 0;
	bool noise_removal = true;
	std::string transformation = "plane";
	double maxxangle = 0.15;
	double maxyangle = 1.0;
	double maxzangle = 0.15;
	double minrad = 2.0;
	double maxrad = 6.0;
	double maxincl = 0.25;
	int winsize = 8;
	int patch_size = 28;
	double min_image_scale = -1.0;
	bool vary_image_pos = false;
	bool random_bg_file = true;
	bool use_lighting = true;
	double albedo_min = 0.4, albedo_max = 0.8;
	double ambient = 0.1;
	double specular_min = 0.3, specular_max = 0.7;
	double shininess_min = 30.0, shininess_max = 150.0;
	int light_temp_min = 3000, light_temp_max = 7000;
	double light_bias_max = 0.0;
	double hue_bias_max = 0.0;
	double saturation_bias_max = 0.0;
	double value_bias_max = 0.0;
	double light_intensity_min = 1.0, light_intensity_max = 4.0;
	double light_dir_dev_max = 0.1;
} param_type;

class SampleMaker
{
public:
	SampleMaker() {}

	param_type params;
	RandGen* random;
	ImageReader imreader, bgreader;
	InfoWriter* infowriter;

	bool ViewAngleTransform(cv::Mat src, cv::Mat dst, double xangle, double yangle, double zangle, int dist);
	void ApplyHSVBias(cv::Mat& image, cv::Vec3d bias);
	bool TransformImage(const cv::Mat& image);
	void PlaceTransformedImage(cv::Mat& bg);
	void ProcessSample();
	void Visualize(std::string imagename);

private:
	cv::Mat img_src, img_tran, alpha;

	std::string out_file_name;
	double maxscale, scale;
	double obj_scale;
	int patch_width, patch_height;
	int obj_width, obj_height;
	double bg_color, fill_color;
	double transparent_color_low, transparent_color_high;
	int x, y;
	int obj_x = 0, obj_y = 0;
	double r;
	double i;
	double xangle, yangle, zangle;
	double light_x, light_y, light_z;
	int light_temp;
	cv::Vec3d light_color;
	cv::Vec3d light_bias;
	cv::Vec3d hsv_bias;
	double light_intensity;
	double object_albedo;
	double object_Ka, object_Kd, object_Ks;
	double object_shininess;
	std::string cascade;
};
