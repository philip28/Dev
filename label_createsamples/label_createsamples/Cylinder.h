#pragma once

#include "Object.h"

#if defined HAVE_TBB
#define NOMINMAX
#include "tbb/tbb.h"
#include "tbb/concurrent_vector.h"
#undef NOMINMAX
#endif

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
	void ProjectImage(cv::Mat src, double rad = -1, double i = 0, double tclow = -1, double tchigh = -1) override
	{
		int cn = src.channels();
		if (src.depth() != CV_8U || !(cn == 3 || cn == 1) || src.dims != 2)
		{
			CV_Error(CV_StsBadArg, "Source must be a two-dimensional array.");
			throw;
		}

		int cx = src.cols / 2;

		for (auto y = 0; y < src.rows; y++)
		{
			double y1 = y;
			double dz = 0;
			double r = rad;
			// inclination
			if (i) {
				dz = y * sin(i);
				r = rad + dz;
				y1 = y * cos(i);
			}

			for (auto l = 0; l < src.cols; l++)
			{
				cv::Vec3d color_rgb;
				double color_gs;

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

				point3d point;
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

		if (i)
			radius = MAX((src.rows - 1) * sin(i) + rad, rad);
		else
			radius = rad;
//		radius = MAX(rad, r);
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