#pragma once

#include "Object.h"

inline void _getcylxz(double r, double phi, int l, double &x, double &z)
{
	double alpha;

	alpha = l / r;
	x = r * sin(phi + alpha);
	z = r * (1 - cos(alpha));
}

class Cylinder : public Object
{
public:
	void ProjectImage(cv::Mat src, double rad = -1, double i = 0, double tclow = -1, double tchigh = -1)
	{
		double r, dz = 0, y1;
		cv::Vec3d color_rgb;
		double color_gs;
		int cx, l, y;
		point3d point;
		const double RW = 0.2126, GW = 0.7152, BW = 0.0722;
		int cn = src.channels();

		if (src.depth() != CV_8U || !(cn == 3 || cn == 1) || src.dims != 2)
		{
			CV_Error(CV_StsBadArg, "Source must be a two-dimensional array.");
			throw;
		}

		cx = src.cols / 2;
		r = rad;

		for (y = 0; y < src.rows; y++)
		{
			y1 = y;
			// inclination
			if (i) {
				dz = y * sin(i);
				r = rad + dz;
				y1 = y * cos(i);
			}

			for (l = 0; l < src.cols; l++)
			{
				if (cn == 3)
				{
					color_rgb = src.at<cv::Vec3b>(y, l);
					color_rgb /= 255.0;
					color_gs = color_rgb[0] * RW + color_rgb[1] * BW + color_rgb[2] * GW;
				}
				else
				{
					color_gs = src.at<uchar>(y, l) / 255.0;
					color_rgb = cv::Vec3d(color_gs, color_gs, color_gs);
				}

				_getcylxz(r, 0, l - cx, point.coords.x, point.coords.z);
				point.coords.z -= dz;
				point.coords.y = y1;
				point.value = color_rgb;

				// check if transparent
				bool tr = tclow >= 0 && tchigh >= tclow && (color_gs >= tclow && color_gs <= tchigh);
				if (tr)
					bg_points.push_back(point);
				else
					points.push_back(point);
			}
		}

		radius = MAX(rad, r);
		GetBounds();

		Shift(0, -maxy / 2, -radius); // optimize later
		center = cv::Vec3d(0, 1, 0);
	};

	virtual cv::Vec3d Normal(const vec3d &P) const override
	{
		cv::Vec3d point(P.x, P.y, P.z);
		double pr_len = point.dot(center);
		cv::Vec3d pr_point = pr_len * center;
		return cv::normalize(point - pr_point);
	}

	double radius = 0;
};