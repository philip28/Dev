#pragma once

#include <string.h>
#include <string>
#include "opencv2/opencv.hpp"
#include <math.h>
#include "ImageTransformData.h"

void CreateTestSamples(std::string imagename, ImageTransformData* data);
void Visualize(std::string imagename, ImageTransformData* data);

bool TransformImage(std::string imagename, ImageTransformData* data);
bool PlaceTransformedImage(cv::Mat background, ImageTransformData* data);
bool ViewAngleTransform(cv::Mat src, cv::Mat dst, double xangle, double yangle, double zangle, int dist);