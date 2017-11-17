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
	void ProjectImage8(cv::Mat src, double rad = -1, double i = 0, double bgcolor = -1)
	{
		double r, dz = 0, y1;
		double color;
		int cx, l, y;
		point3d point;

		if (src.depth() != CV_8U || src.channels() != 1 || src.dims != 2)
		{
			CV_Error(CV_StsBadArg, "Source must be a two-dimensional array of CV_8UC1 type.");
			throw;
		}

		//		cx = (double)src.cols / 2;
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
				color = src.at<uchar>(y, l) / 255.0;
				// check if transparent
				if (bgcolor >= 0 && color != bgcolor)
				{
					_getcylxz(r, 0, l - cx, point.coords.x, point.coords.z);
					point.coords.z -= dz;
					point.coords.y = y1;
					point.value[0] = point.value[1] = point.value[2] = color;
					points.push_back(point);
				}
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