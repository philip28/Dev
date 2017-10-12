#pragma once

#include <string>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class BackgroundImageReader
{
public:
	bool Create(string filename);
	int Next();
	int NextRandom();

	int count;
	int pos;
	int index;
	cv::Mat image;
	string imagename;
	vector<string> filelist;
};