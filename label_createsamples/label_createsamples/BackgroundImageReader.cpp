#include "BackgroundImageReader.h"
#include <fstream>

bool BackgroundImageReader::Create(string filename)
{
	string line;

	ifstream file(filename.c_str());
	if (!file.is_open())
		return false;

	count = 0;
	pos = 0;
	index = 0;
	while (!file.eof())
	{
		getline(file, line);
		line.erase(line.find_last_not_of(" \n\r\t") + 1);
		if (line.empty()) continue;
		if (line.at(0) == '#') continue; /* comment */
		filelist.push_back(line);
		count++;
	}

	return true;
}

int BackgroundImageReader::Next()
{
	if (index >= count) return -1;

	imagename = filelist[index++];
	image = imread(imagename.c_str(), IMREAD_GRAYSCALE);
	if (image.empty()) {
		CV_Error(CV_StsBadArg, "Error opening background image");
		return -1;
	}
	pos++;

	return pos;
}

int BackgroundImageReader::NextRandom()
{
	if (index >= count) return -1;

	index++;
	pos = rand() % count;
	pos %= count;
	imagename = filelist[pos];
	image = imread(imagename.c_str(), IMREAD_GRAYSCALE);
	if (image.empty()) {
		CV_Error(CV_StsBadArg, "Error opening background image");
		return -1;
	}

	return pos;
}