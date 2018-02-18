#pragma once

#include <string>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

class BackgroundImageReader
{
public:
	bool Create(std::string filename, bool gs = false)
	{
		boost::filesystem::path p(filename);

		if (!boost::filesystem::exists(p))
			return false;

		count = 0;
		pos = 0;
		index = 0;

		if (boost::filesystem::is_regular_file(p))
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
				count++;
			}
		}
		else
		{
			for (auto&& x : boost::filesystem::directory_iterator(p))
			{
				if (boost::filesystem::is_regular_file(x.path()))
				{
					filelist.push_back(x.path().string());
					count++;
				}
			}
		}

		grayscale = gs;

		return true;
	}

	int Next()
	{
		if (index >= count) return -1;

		imagefullname = filelist[index++];
		ExtractFileName();

		if (grayscale)
			image = cv::imread(imagefullname.c_str(), cv::IMREAD_GRAYSCALE);
		else
			image = cv::imread(imagefullname.c_str());
		if (image.empty()) {
			CV_Error(CV_StsBadArg, "Error opening background image " + imagefullname);
			return -1;
		}
		pos++;

		return pos;
	}

	int NextRandom()
	{
		if (index >= count) return -1;

		index++;
		pos = (int)((filelist.size() - 1) * ((double)rand() / RAND_MAX));
		imagefullname = filelist[pos];
		ExtractFileName();

		if (grayscale)
			image = cv::imread(imagefullname.c_str(), cv::IMREAD_GRAYSCALE);
		else
			image = cv::imread(imagefullname.c_str());
		if (image.empty()) {
			CV_Error(CV_StsBadArg, "Error opening background image");
			return -1;
		}

		return pos;
	}

	int count = 0;
	int pos = 0;
	int index = 0;
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
};