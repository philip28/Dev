#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"

#if defined HAVE_TBB
#define NOMINMAX
#include "tbb/tbb.h"
#undef NOMINMAX
#endif

static const double kInfinity = std::numeric_limits<double>::max();

class Light
{
public:
	Light(const cv::Vec3d c, const cv::Vec3d &i = 1) : intensity(i), color(c) {};
	virtual ~Light() {}
	virtual void Illuminate(const point3d &P, cv::Vec3d &, cv::Vec3d &, double &) const = 0;
	cv::Vec3d color;
	cv::Vec3d intensity;

	static cv::Vec3d ApplyBias(cv::Vec3d src, cv::Vec3d bias)
	{
		cv::Vec3d rgb = src.mul(bias);
		clamp(rgb, 0, 1);
		
		return rgb;
	}

	static cv::Vec3d ColorTempToRGB(unsigned int kelvin, int mode = 0)
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

		cv::Vec3d rgb;
		double temp = kelvin / 100.0;

		if (temp <= 66)
		{
			rgb[R] = 255;

			rgb[G] = temp;
			rgb[G] = 99.4708025861 * std::log(rgb[G]) - 161.1195681661;

			if (temp <= 19)
			{
				rgb[B] = 0;
			}
			else {
				rgb[B] = temp - 10;
				rgb[B] = 138.5177312231 * std::log(rgb[B]) - 305.0447927307;
			}
		}
		else
		{
			rgb[R] = temp - 60;
			rgb[R] = 329.698727446 * std::pow(rgb[R], -0.1332047592);

			rgb[G] = temp - 60;
			rgb[G] = 288.1221695283 * std::pow(rgb[G], -0.0755148492);

			rgb[B] = 255;

		}

		rgb /= 255.0;
		clamp(rgb, 0, 1);

		return rgb;
	}

private:
	static void clamp(cv::Vec3d& x, double min, double max)
	{
		for (int i = 0; i < 3; i++)
		{
			if (x[i] < min) x[i] = min;
			if (x[i] > max) x[i] = max;
		}
	}
};

class DistantLight : public Light
{
	cv::Vec3d dir;
public:
	DistantLight(const cv::Vec3d &P, const cv::Vec3d &c, const cv::Vec3d &i = 1) : Light(c, i), dir(P) {}
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
	PointLight(const cv::Vec3d &P, const cv::Vec3d &c, const cv::Vec3d &i = 1) : Light(c, i), pos(P) {}
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
		lights.push_back(l);
	}
	void AddObject(Object *o)
	{
		objects.push_back(o);
	}

	void Render(Plane &frame, bool depth = true)
	{
		std::vector<point3d>::iterator it, it_end;

		for (int i = 0; i < objects.size(); i++)
		{
			if (depth) objects[i]->ZSort();

			it = objects[i]->points.begin();
			it_end = objects[i]->points.end();
			for (; it != it_end; ++it)
			{
				frame.points.push_back(CastRay(*it, *objects[i]));
			}
			frame.bg_points = objects[i]->bg_points;
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
			specular += std::pow(std::max(0.0, R.dot(viewdir)), o.surface.shininess) * cv::Vec3d(1, 1, 1);// *intensity;
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

	void GetLightingRange(Object &o, cv::Vec3d &lmin, cv::Vec3d &lmax)
	{
		cv::Vec3d viewdir(0, 0, -1);
		cv::Vec3d lx1, lx2, ly1, ly2;
		lx1 = Reflect(-viewdir, o.Normal(o.MinX()->coords));
		lx2 = Reflect(-viewdir, o.Normal(o.MaxX()->coords));
		ly1 = Reflect(-viewdir, o.Normal(o.MinY()->coords));
		ly2 = Reflect(-viewdir, o.Normal(o.MaxY()->coords));
		lmin[0] = std::min(std::min(lx1[0], lx2[0]), std::min(ly1[0], ly2[0]));
		lmin[1] = std::min(std::min(lx1[1], lx2[1]), std::min(ly1[1], ly2[1]));
		lmin[2] = std::min(std::min(lx1[2], lx2[2]), std::min(ly1[2], ly2[2]));
		lmax[0] = std::max(std::max(lx1[0], lx2[0]), std::max(ly1[0], ly2[0]));
		lmax[1] = std::max(std::max(lx1[1], lx2[1]), std::max(ly1[1], ly2[1]));
		lmax[2] = std::max(std::max(lx1[2], lx2[2]), std::max(ly1[2], ly2[2]));
	}

	std::vector<Light*> lights;
	std::vector<Object*> objects;
	cv::Vec3d viewdir;
};