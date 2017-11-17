#pragma once

#include "Object.h"

class Plane : public Object
{
public:
	void ProjectImage8(cv::Mat src, double bgcolor = -1)
	{
		point3d point;
		double color;

		for (int y = 0; y < src.rows; y++)
		{
			for (int x = 0; x < src.cols; x++)
			{
				color = src.at<uchar>(y, x) / 255.0;
				// check if transparent
				if (bgcolor >= 0 && color != bgcolor)
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