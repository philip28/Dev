#pragma once

#include <string>
#include <stdio.h>
#include "ImageTransformData.h"

class InfoWriter
{
public:
	InfoWriter() : fInitialized(false) {}
	~InfoWriter()
	{
		if (fInitialized)
			fclose(file);
	};

	bool Create(string filename)
	{
		const char* dir = NULL;

		file = fopen(filename.c_str(), "w");
		if (file == NULL)
			return false;

		size_t found = filename.rfind('\\');
		if (found == string::npos) {
			found = filename.rfind('/');
		}
		if (found == string::npos) {
			folder = "";
		}
		else {
			folder = filename.substr(0, found);
		}

		fInitialized = true;

		return true;
	}

	bool WriteInfo(int num, ImageTransformData* data, string ext = ".jpg")
	{
		char imagename[_MAX_PATH];

		sprintf(imagename, "%s.%04d_%04d_%04d_%04d_%04d%s", data->bgname.c_str(), num, data->x, data->y, data->width, data->height, ext.c_str());
		fprintf(file, "%s %d %d %d %d %d\n", imagename, 1, data->x, data->y, data->width, data->height);
		return true;
	}

	bool WriteImage(int num, ImageTransformData* data, Mat image, string ext = ".jpg")
	{
		char imagename[_MAX_PATH];
		string fullpath;

		sprintf(imagename, "%s.%04d_%04d_%04d_%04d_%04d%s", data->bgname.c_str(), num, data->x, data->y, data->width, data->height, ext.c_str());
		fullpath = folder + '\\' + imagename;
		imwrite(fullpath.c_str(), image);
		return true;
	}

private:
	bool fInitialized;
	FILE* file;
	string folder;
};

