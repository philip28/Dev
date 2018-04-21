#pragma once

#include <string>
#include <fstream>
#include "boost/filesystem.hpp"
#include "RandGen.h"

#if defined HAVE_TBB
#define NOMINMAX
#include "tbb/tbb.h"
#undef NOMINMAX
#endif

class InfoWriter
{
public:
	~InfoWriter()
	{
		if (info_file.is_open())
			info_file.close();
	};

	bool Init(std::string o, std::string i, std::string bg, RandGen* r)
	{
		output_dir = boost::filesystem::path(o);
		info_name = boost::filesystem::path(i);
		bg_output_dir = boost::filesystem::path(bg);

		if (!boost::filesystem::exists(output_dir))
			boost::filesystem::create_directory(output_dir);

		if (!bg.empty())
			if (!boost::filesystem::exists(bg_output_dir))
				boost::filesystem::create_directory(bg_output_dir);

		if (!i.empty())
		{
			boost::filesystem::path fullname = output_dir;
			fullname /= info_name;
			info_file = boost::filesystem::ofstream(fullname, std::ios::out);
			if (!info_file.is_open())
				return false;
		}

		random = r;

		return true;
	}

	bool WriteInfo(std::string out_file_name, int x, int y, int width, int height, std::string ext = ".jpg") // obsolete
	{
		if (!info_file.is_open())	// No info file requested, so do nothing
			return true;
		
		std::string filename = MakeFileName(out_file_name, width, height, ext);
#if defined HAVE_TBB
		tbb::spin_mutex::scoped_lock lock(mutex);
#endif
		info_file << filename << " " << 1 << " " << x << " " << y << " " << width << " " << height << std::endl;
		return true;
	}

	std::string MakeFileName(std::string out_file_name, int width, int height, std::string ext = ".jpg")
	{
		std::string prefix;
		if (out_file_name.size())
			prefix = out_file_name + ".";

		static std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
		std::string unique_id;
		unique_id.resize(8);
		for (int i = 0; i < 8; i++)
			unique_id[i] = charset[random->InRangeI(0, (int)charset.length() - 1)];

		std::string filename;
		filename = prefix + std::to_string(width) + "x" + std::to_string(height) + "." + unique_id + ext;

		return filename;
	}

	bool WriteImageEx(std::string out_file_name, cv::Mat image, int width, int height, std::string ext = ".jpg")
	{
		std::string filename = MakeFileName(out_file_name, width, height, ext);
		boost::filesystem::path fullpath = output_dir;
		fullpath /= boost::filesystem::path(filename);
		imwrite(fullpath.string(), image);
		return true;
	}

	bool WriteImage(std::string out_file_name, std::string out_dir, cv::Mat image)
	{
		boost::filesystem::path fullpath = out_dir;
		fullpath /= boost::filesystem::path(out_file_name);
		imwrite(fullpath.string(), image);
		return true;
	}

private:
	std::ofstream info_file;
	boost::filesystem::path output_dir;
	boost::filesystem::path info_name;
	boost::filesystem::path bg_output_dir;
	RandGen* random;

#if defined HAVE_TBB
	tbb::spin_mutex mutex;
#endif
};

