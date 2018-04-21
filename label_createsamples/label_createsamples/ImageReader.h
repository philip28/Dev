#pragma once

#include <string>
#include <algorithm>
#include "RandGen.h"
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

#if defined HAVE_TBB
#define NOMINMAX
#include "tbb/tbb.h"
#undef NOMINMAX
#endif

enum input_type
{
	IMAGE,
	DIR,
	LIST
};

class ImageReader
{
public:
	bool Init(std::string filename, int type, RandGen* r, bool gs = false)
	{
		boost::filesystem::path p(filename);

		if (!boost::filesystem::exists(p))
			return false;

		rec_count = 0;
		rec_pos = 0;
		rec_index = 0;

		if (type == input_type::LIST)
		{
			std::string line;

			std::ifstream file(filename.c_str());
			if (!file.is_open())
				return false;

			while (!file.eof())
			{
				getline(file, line);
				line.erase(line.find_last_not_of(" \n\r\t") + 1);
				if (line.empty()) continue;
				if (line.at(0) == '#') continue; /* comment */
				filelist.push_back(line);
				rec_count++;
			}
		}
		else if (type == input_type::IMAGE)
		{
			filelist.push_back(filename);
			rec_count++;
		}
		else if (type == input_type::DIR)
		{
			for (auto&& x : boost::filesystem::directory_iterator(p))
			{
				if (boost::filesystem::is_regular_file(x.path()))
				{
					filelist.push_back(x.path().string());
					rec_count++;
				}
			}
		}
		else
			return false;

		grayscale = gs;
		random = r;

		return true;
	}

	size_t Get(size_t i)
	{
		if (i >= filelist.size())
		{
			CV_Error(CV_StsBadArg, "Index out of range");
			return -1;
		}

		imagefullname = filelist[i];
		ExtractFileName();

		if (grayscale)
			image = cv::imread(imagefullname.c_str(), cv::IMREAD_GRAYSCALE);
		else
			image = cv::imread(imagefullname.c_str());
		if (image.empty()) {
			CV_Error(CV_StsBadArg, "Error opening background image " + imagefullname);
			return -1;
		}

		return i;
	}

	size_t GetRandom()
	{
		rec_pos = random->InRangeI(0, (int)filelist.size() - 1);
		rec_pos = Get(rec_pos);
		return rec_pos;
	}

	size_t Next(bool r=false)
	{
		if (rec_index >= rec_count) return -1;

		if (r)
			rec_pos = GetRandom();
		else
			rec_pos = Get(rec_pos);

		rec_index++;
		rec_pos++;

		return rec_pos-1;
	}

	void Rewind()
	{
		rec_index = 0;
		rec_pos = 0;
	}

	size_t rec_count = 0;
	size_t rec_pos = 0;
	size_t rec_index = 0;
	cv::Mat image;
	std::string imageshortname, imagefullname;
	std::vector<std::string> filelist;

private:
	void ExtractFileName()
	{
		size_t found = imagefullname.rfind('\\');
		if (found == std::string::npos) {
			found = imagefullname.rfind('/');
		}
		if (found == std::string::npos) {
			imageshortname = imagefullname;
		}
		else {
			imageshortname = imagefullname.substr(found + 1, imagefullname.length() - found);
		}

		found = imageshortname.rfind('.');
		if (found != std::string::npos) {
			imageshortname = imageshortname.substr(0, found);
		}

		std::replace(imageshortname.begin(), imageshortname.end(), ' ', '_');
	}

	bool grayscale = false;
	RandGen* random;
};