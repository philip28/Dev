#pragma once

#include <string>
#include <fstream>
#include "boost/filesystem.hpp"
#include "ImageTransformData.h"

class InfoWriter
{
public:
	InfoWriter() {}
	~InfoWriter()
	{
		if (info_file.is_open())
			info_file.close();
	};

	bool Create(std::string o, std::string i)
	{
		output_dir = boost::filesystem::path(o);
		info_name = boost::filesystem::path(i);

		if (!boost::filesystem::exists(output_dir))
			boost::filesystem::create_directory(output_dir);

		if (!i.empty())
		{
			boost::filesystem::path fullname = output_dir;
			fullname /= info_name;
			info_file = boost::filesystem::ofstream(fullname, std::ios::out);
			if (!info_file.is_open())
				return false;
		}

		return true;
	}

	bool WriteInfo(int num, ImageTransformData* data, std::string ext = ".jpg")
	{
		if (!info_file.is_open())	// No info file requested, so do nothing
			return true;
		
		char imagename[_MAX_PATH];

		sprintf(imagename, "%s.%04d_%04d_%04d_%04d_%04d%s", data->bgname.c_str(), num, data->x, data->y, data->width, data->height, ext.c_str());
		info_file << imagename << " " << 1 << " " << data->x << " " << data->y << " " << data->width << " " << data->height << std::endl;
		return true;
	}

	bool WriteImage(int num, ImageTransformData* data, cv::Mat image, std::string ext = ".jpg")
	{
		char imagename[_MAX_PATH];

		sprintf(imagename, "%s.%04d_%04d_%04d_%04d_%04d%s", data->bgname.c_str(), num, data->x, data->y, data->width, data->height, ext.c_str());
		boost::filesystem::path fullpath = output_dir;
		fullpath /= boost::filesystem::path(imagename);
		imwrite(fullpath.string(), image);
		return true;
	}

private:
	std::ofstream info_file;
	boost::filesystem::path output_dir;
	boost::filesystem::path info_name;
};

