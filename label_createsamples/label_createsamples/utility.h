#pragma once

#include <string.h>
#include <string>
#include "opencv2/opencv.hpp"
#include <math.h>
#include "ImageTransformData.h"

using namespace std;
using namespace cv;

void CreateTestSamples(string infoname,
	string imagename,
	string bgname,
	ImageTransformData* data);

bool TransformImage(string imagename, ImageTransformData* data);

bool PlaceTransformedImage(Mat background, ImageTransformData* data);

bool ViewAngleTransform(Mat src, Mat dst, double xangle, double yangle, double zangle, int dist);