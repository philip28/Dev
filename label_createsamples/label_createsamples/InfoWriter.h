#pragma once

#include <string>
#include <stdio.h>
#include "ImageTransformData.h"

using namespace std;

class InfoWriter
{
public:
	InfoWriter();
	~InfoWriter();

	bool Create(string filename);
	bool WriteInfo(int num, ImageTransformData* data);
	bool WriteImage(int num, ImageTransformData* data, Mat image);

private:
	bool fInitialized;
	FILE* file;
	string folder;
};

