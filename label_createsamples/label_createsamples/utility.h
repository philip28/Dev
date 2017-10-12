#pragma once

#include <string.h>
#include <string>
#include "opencv2/opencv.hpp"
#include <math.h>
#include "ImageTransformData.h"

using namespace std;
using namespace cv;

#define PI 3.1415926535

void CreateTestSamples(string infoname,
	string imagename,
	string bgname,
	ImageTransformData* data);

bool TransformImage(string imagename, ImageTransformData* data);

bool PlaceTransformedImage(Mat background, ImageTransformData* data);

bool WrapCylinderTransform(Mat src, Mat dst, double r, double phi);
bool ViewAngleTransform(Mat src, Mat dst, double xangle, double yangle, double zangle, int dist);