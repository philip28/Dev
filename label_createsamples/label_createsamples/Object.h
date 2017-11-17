#pragma once

#include <vector>
#include "opencv2/opencv.hpp"
//#include "algebra.h"

#define PI 3.1415926535

typedef struct
{
	unsigned char R, G, B;
} RGB;

typedef struct
{
	double x, y, z;
} vec3d;

typedef struct
{
	vec3d coords;
	cv::Vec3d value;
} point3d;

typedef std::vector<point3d> point3d_vec;

class Object
{
public:
	void Shift(double x, double y, double z)
	{
		Mat M = (Mat_<double>(4, 4) <<
			1, 0, 0, x,
			0, 1, 0, y,
			0, 0, 1, z,
			0, 0, 0, 1);

		Transform(M);
	};

	void Rotate(double xangle, double yangle, double zangle)
	{
		// Rotate around X axis
		Mat RX = (Mat_<double>(4, 4) <<
			1, 0, 0, 0,
			0, cos(xangle), sin(xangle), 0,
			0, -sin(xangle), cos(xangle), 0,
			0, 0, 0, 1);

		// Rotate around Y axis
		Mat RY = (Mat_<double>(4, 4) <<
			cos(yangle), 0, sin(yangle), 0,
			0, 1, 0, 0,
			-sin(yangle), 0, cos(yangle), 0,
			0, 0, 0, 1);

		// Rotate around Z axis
		Mat RZ = (Mat_<double>(4, 4) <<
			cos(zangle), sin(zangle), 0, 0,
			-sin(zangle), cos(zangle), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		// Composed matrix M->RY->RX->RZ
		Mat R = RZ * RX * RY;

		Transform(R);
		// TODO: Add some approximation to eliminate rotation artefacts
	};


	void Transform(cv::Mat m)
	{
		std::vector<point3d>::iterator it, it_end;
		double x, y, z;

		it = points.begin();
		it_end = points.end();
		for (; it != it_end; ++it)
		{
			x = m.at<double>(0, 0) * it->coords.x + m.at<double>(0, 1) * it->coords.y + m.at<double>(0, 2) * it->coords.z + m.at<double>(0, 3);
			y = m.at<double>(1, 0) * it->coords.x + m.at<double>(1, 1) * it->coords.y + m.at<double>(1, 2) * it->coords.z + m.at<double>(1, 3);
			z = m.at<double>(2, 0) * it->coords.x + m.at<double>(2, 1) * it->coords.y + m.at<double>(2, 2) * it->coords.z + m.at<double>(2, 3);
			it->coords.x = x;
			it->coords.y = y;
			it->coords.z = z;
		}

		x = m.at<double>(0, 0) * center[0] + m.at<double>(0, 1) * center[1] + m.at<double>(0, 2) * center[2] + m.at<double>(0, 3);
		y = m.at<double>(1, 0) * center[0] + m.at<double>(1, 1) * center[1] + m.at<double>(1, 2) * center[2] + m.at<double>(1, 3);
		z = m.at<double>(2, 0) * center[0] + m.at<double>(2, 1) * center[1] + m.at<double>(2, 2) * center[2] + m.at<double>(2, 3);
		center[0] = x;
		center[1] = y;
		center[2] = z;
		center = cv::normalize(center);

		GetBounds();
	}

	void OutMat(cv::Mat &m, unsigned int bgcolor, bool depth = true)
	{
		GetBounds();

		int dst_width, dst_height;
		std::vector<point3d>::iterator it, it_end;

		// initialize dest image
		dst_width = (int)abs(round(maxx - minx)) + 1;
		dst_height = (int)abs(round(maxy - miny)) + 1;
		m.create(dst_height, dst_width, CV_8UC1);
		m.setTo(cv::Scalar(bgcolor));

		// Z-buffer filtering
		if (depth)
			std::sort(points.begin(), points.end(), [](point3d a, point3d b) {
				return a.coords.z > b.coords.z;
			});

		// Only grayscale so far (assumed R)
		it = points.begin();
		it_end = points.end();
		for (; it != it_end; ++it) {
			uchar tc = (uchar)std::max(0.0, std::min(it->value[0], 1.0) * 255);
			m.at<uchar>((int)(it->coords.y - miny), (int)(it->coords.x - minx)) = (uchar)std::max(0.0, std::min(it->value[0], 1.0) * 255);
		}
	}

	double MinX()
	{
		return (*std::min_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.x < b.coords.x;
		})).coords.x;
	}
	double MaxX()
	{
		return (*std::max_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.x < b.coords.x;
		})).coords.x;
	}
	double MinY()
	{
		return (*std::min_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.y < b.coords.y;
		})).coords.y;
	}
	double MaxY()
	{
		return (*std::max_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.y < b.coords.y;
		})).coords.y;
	}

	void GetBounds()
	{
		maxx = MaxX();
		minx = MinX();
		maxy = MaxY();
		miny = MinY();
	}

	virtual cv::Vec3d Normal(const vec3d &) const = 0;


	point3d_vec points;
	cv::Vec3d center;

	struct {
		cv::Vec3d albedo;
		cv::Vec3d Ka; // phong model ambient reflectivity coeeficient (RGB)
		cv::Vec3d Kd; // phong model diffuse eflectivity coeeficient (RGB)
		cv::Vec3d Ks; // phong model specular eflectivity coeeficient (RGB)
		double shininess = 20; // phong specular exponent
	} surface;

	double minx = -1, miny = -1, maxx = -1, maxy = -1;
};