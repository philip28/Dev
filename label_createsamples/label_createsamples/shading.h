#pragma once

//#include "algebra.h"
#include <algorithm>
#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"

static const double kInfinity = std::numeric_limits<double>::max();

class Light
{
public:
	Light(const cv::Vec3d &c = 1, const cv::Vec3d &i = 1) : color(c), intensity(i) {}
	virtual ~Light() {}
	virtual void Illuminate(const point3d &P, cv::Vec3d &, cv::Vec3d &, double &) const = 0;
	cv::Vec3d color;
	cv::Vec3d intensity;
};

class DistantLight : public Light
{
	cv::Vec3d dir;
public:
	DistantLight(const cv::Vec3d &P, const cv::Vec3d &c = 1, const cv::Vec3d &i = 1) : Light(c, i), dir(P) {}
	void Illuminate(const point3d &P, cv::Vec3d &LightDir, cv::Vec3d &LightIntensity, double &Distance) const override
	{
		LightDir = -dir; // Surface to light
		LightIntensity = P.value.mul(color.mul(intensity));
		Distance = kInfinity;
	}
};

class PointLight : public Light
{
	cv::Vec3d pos;
public:
	PointLight(const cv::Vec3d &P, const cv::Vec3d &c = 1, const cv::Vec3d &i = 1) : Light(c, i), pos(P) {}
	void Illuminate(const point3d &P, cv::Vec3d &LightDir, cv::Vec3d &LightIntensity, double &Distance) const override
	{
		cv::Vec3d hitpoint(P.coords.x, P.coords.y, P.coords.z);
		LightDir = hitpoint - pos;
		double r2 = cv::norm(LightDir);
		LightDir = cv::normalize(LightDir);
		LightIntensity = P.value * intensity / (4 * PI * r2 * r2);
	}
};

class Scene
{
public:
	Scene() : viewdir(0, 0, -1) {}

	void AddLight(Light *l)
	{
		lights.push_back(std::unique_ptr<Light>(l));
	}
	void AddObject(Object *o)
	{
		objects.push_back(std::unique_ptr<Object>(o));
	}

	void Render(Plane &frame)
	{
		std::vector<point3d>::iterator it, it_end;

		for (int i = 0; i < objects.size(); i++)
		{
			it = objects[i]->points.begin();
			it_end = objects[i]->points.end();
			for (; it != it_end; ++it)
			{
				frame.points.push_back(CastRay(*it, *objects[i]));
			}
		}
	}

	point3d CastRay(const point3d &P, const Object &o)
	{
		cv::Vec3d hitcolor;
		point3d hitpoint;
		cv::Vec3d lightdir, intensity, normal;
		cv::Vec3d ambient = 0, diffuse = 0, specular = 0;
		double distance;

		normal = o.Normal(P.coords);

		for (int i = 0; i < lights.size(); i++)
		{
			lights[i]->Illuminate(P, lightdir, intensity, distance);
			cv::Vec3d R = Reflect(lightdir, normal);

			ambient += intensity;
			diffuse += std::max(0.0, normal.dot(lightdir)) * intensity;
			double rv = R.dot(viewdir);
			specular += std::pow(std::max(0.0, R.dot(viewdir)), o.surface.shininess) * Vec3d(1, 1, 1);// *intensity;
		}
		
		hitpoint.value = o.surface.Ka.mul(ambient) + o.surface.Kd.mul(o.surface.albedo.mul(diffuse)) + o.surface.Ks.mul(specular);

		hitpoint.coords.x = P.coords.x;
		hitpoint.coords.y = P.coords.y;
		hitpoint.coords.z = 0;

		return hitpoint;
	}

	cv::Vec3d Reflect(const cv::Vec3d &I, const cv::Vec3d &N)
	{
		return -I + 2 * I.dot(N) * N;
	}

	std::vector<std::unique_ptr<Light>> lights;
	std::vector<std::unique_ptr<Object>> objects;
	cv::Vec3d viewdir;
};