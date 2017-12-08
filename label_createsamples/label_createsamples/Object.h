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

	void OutMat(cv::Mat &m, int type = CV_8U, cv::Vec3d bgcolor = 0, bool depth = true, bool square = false, bool approx = false)
	{
		GetBounds();

		int dst_width, dst_height;
		std::vector<point3d>::iterator it, it_end;

		// initialize dest image
		dst_width = (int)abs(round(maxx - minx)) + 1;
		dst_height = (int)abs(round(maxy - miny)) + 1;
		int dx = 0, dy = 0;

		if (square)
		{
			dx = (std::max(dst_width, dst_height) - dst_width) / 2;
			dy = (std::max(dst_width, dst_height) - dst_height) / 2;
			dst_width = dst_height = std::max(dst_width, dst_height);
		}

		cv::Vec3i bgc = 255 * bgcolor;

		if (type == CV_8U)
		{
			m.create(dst_height, dst_width, CV_8UC2); // with alpha channel
			if (bgc[0] >= 0)
				if (!approx)
					m.setTo(cv::Scalar(bgc[0], 255));
				else
					m.setTo(cv::Scalar(bgc[0], 0));
			else
				m.setTo(cv::Scalar(0, 0));
		}
		else if (type == CV_32S)
		{
			m.create(dst_height, dst_width, CV_8UC4); // with alpha channel
			if (bgc[0] >= 0)
				if (!approx)
					m.setTo(cv::Scalar(bgc[0], bgc[1], bgc[2], 255));
				else
					m.setTo(cv::Scalar(bgc[0], bgc[1], bgc[2], 0));
			else
				m.setTo(cv::Scalar(0, 0, 0, 0));
		}

		// Z-buffer filtering
		if (depth)
			ZSort();

		it = points.begin();
		it_end = points.end();

		if (type == CV_8U)
		{
			for (; it != it_end; ++it) {
				uchar tc = (uchar)std::max(0.0, std::min(it->value[0], 1.0) * 255);
				m.at<Vec2b>((int)(it->coords.y - miny) + dy, (int)(it->coords.x - minx) + dx)[0] = tc;
				m.at<Vec2b>((int)(it->coords.y - miny) + dy, (int)(it->coords.x - minx) + dx)[1] = 255;
			}
		}
		else if (type == CV_32S)
		{
			for (; it != it_end; ++it) {
				cv::Vec3b tc;
				tc[0] = (uchar)std::max(0.0, std::min(it->value[0], 1.0) * 255);
				tc[1] = (uchar)std::max(0.0, std::min(it->value[1], 1.0) * 255);
				tc[2] = (uchar)std::max(0.0, std::min(it->value[2], 1.0) * 255);
				m.at<cv::Vec4b>((int)(it->coords.y - miny) + dy, (int)(it->coords.x - minx) + dx) = cv::Scalar(tc);
				m.at<cv::Vec4b>((int)(it->coords.y - miny) + dy, (int)(it->coords.x - minx) + dx)[3] = 255;
			}
		}

		// Artefact removal
		if (approx)
		{
			if (type == CV_8U)
			{
				for (int x = 1; x < (m.cols - 1); x++)
				{
					for (int y = 1; y < (m.rows - 1); y++)
					{
						if (m.at<Vec2b>(y, x)[1] == 0)
						{
							if (m.at<Vec2b>(y - 1, x)[1] && m.at<Vec2b>(y + 1, x)[1] && m.at<Vec2b>(y, x + 1)[1] && m.at<Vec2b>(y, x - 1)[1]) {
								m.at<Vec2b>(y, x)[0] = (m.at<Vec2b>(y - 1, x)[0] + m.at<Vec2b>(y - 1, x)[0] + m.at<Vec2b>(y + 1, x)[0] + m.at<Vec2b>(y, x + 1)[0]) / 4;
								m.at<Vec2b>(y, x)[1] = 255;
							}
							if (bgc[0] >= 0) m.at<Vec2b>(y, x)[1] = 255;
						}
					}
				}
							
			}
			else if (type == CV_32S)
			{
				for (int x = 1; x < (m.cols - 1); x++)
				{
					for (int y = 1; y < (m.rows - 1); y++)
					{
						if (m.at<Vec4b>(y, x)[3] == 0)
						{
							if (m.at<Vec4b>(y - 1, x)[3] && m.at<Vec4b>(y + 1, x)[3] && m.at<Vec4b>(y, x + 1)[3] && m.at<Vec4b>(y, x - 1)[3]) {
								m.at<Vec3b>(y, x) = (m.at<Vec3b>(y - 1, x) + m.at<Vec3b>(y - 1, x) + m.at<Vec3b>(y + 1, x) + m.at<Vec3b>(y, x + 1)) / 4;
								m.at<Vec4b>(y, x)[3] = 255;
							}
							if (bgc[0] >= 0) m.at<Vec2b>(y, x)[3] = 255;
						}
					}
				}
			}
		}
	}

	static void RemoveAlpha(const cv::Mat &src, cv::Mat &dst)
	{
		if (src.channels() != 2 && src.channels() != 4)
		{
			CV_Error(CV_StsBadArg, "Source must be an array with alpha channel.");
			throw;
		}

		if (src.channels() == 2)
		{
			cv::Mat chans[2];
			cv::split(src, chans);
			dst = chans[0].clone();
		}
		else
		{
			cv::Mat chans[4];
			cv::split(src, chans);
			cv::merge(chans, 3, dst);
		}
	}

	void ZSort()
	{
		std::sort(points.begin(), points.end(), [](point3d a, point3d b) {
		return a.coords.z > b.coords.z;
		});
	}

	point3d* MinX()
	{
		return &(*std::min_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.x < b.coords.x;
		}));
	}
	point3d* MaxX()
	{
		return &(*std::max_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.x < b.coords.x;
		}));
	}
	point3d* MinY()
	{
		return &(*std::min_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.y < b.coords.y;
		}));
	}
	point3d* MaxY()
	{
		return &(*std::max_element(points.begin(), points.end(), [](point3d a, point3d b) {
			return a.coords.y < b.coords.y;
		}));
	}

	void GetBounds()
	{
		maxx = MaxX()->coords.x;
		minx = MinX()->coords.x;
		maxy = MaxY()->coords.y;
		miny = MinY()->coords.y;
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