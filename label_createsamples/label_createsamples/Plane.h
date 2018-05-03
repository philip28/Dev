#pragma once

#include "Object.h"

class Plane : public Object
{
public:
	void ProjectImage(cv::Mat src, double rad = -1, double i = 0, double tclow = -1, double tchigh = -1, int mode = 0) override
	{
		int R, G, B;
		if (mode == 0)
		{
			// BGR, opencv
			R = 2;
			G = 1;
			B = 0;
		}
		else
		{
			R = 0;
			G = 1;
			B = 2;
		}

		int cn = src.channels();

		for (int y = 0; y < src.rows; y++)
		{
			for (int x = 0; x < src.cols; x++)
			{
				cv::Vec3d color_rgb;
				double color_gs;

				if (cn == 3)
				{
					color_rgb = src.at<cv::Vec3b>(y, x);
					color_rgb /= 255.0;
					color_gs = color_rgb[R] * RW + color_rgb[G] * GW + color_rgb[B] * BW;
				}
				else
				{
					color_gs = src.at<uchar>(y, x) / 255.0;
					color_rgb = cv::Vec3d(color_gs, color_gs, color_gs);
				}

				point3d point;
				point.coords.x = x;
				point.coords.y = y;
				point.coords.z = 0;
				point.value = color_rgb;

				// check if transparent
				bool tr = tclow >= 0 && tchigh >= tclow && (color_gs >= tclow && color_gs <= tchigh);
				if (tr)
					bg_points.push_back(point);
				else
					points.push_back(point);
			}
		}

		GetBounds();
		Shift(-maxx / 2, -maxy / 2, 0); // optimize later

		center = cv::Vec3d(0, 0, -1);
	};
	void ProjectImage8(cv::Mat src, double tclow = -1, double tchigh = -1)
	{
		point3d point;
		double color;

		for (int y = 0; y < src.rows; y++)
		{
			for (int x = 0; x < src.cols; x++)
			{
				color = src.at<uchar>(y, x) / 255.0;
				// check if transparent
				bool tr = tclow >= 0 && tchigh >= tclow && (color >= tclow && color <= tchigh);
				if (!tr)
				{
					point.coords.x = x;
					point.coords.y = y;
					point.coords.z = 0;
					point.value[0] = point.value[1] = point.value[2] = color;
					points.push_back(point);
				}
			}
		}

		GetBounds();
		Shift(-maxx / 2, -maxy / 2, 0); // optimize later

		center = cv::Vec3d(0, 0, -1);
	};

	virtual cv::Vec3d Normal(const vec3d &P) const override
	{
		return (center);
	}

	double radius = 0;
};