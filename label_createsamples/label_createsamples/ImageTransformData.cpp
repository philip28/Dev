#include "ImageTransformData.h"

ImageTransformData::ImageTransformData()
{
	params.numsample = 1000;
	params.bgcolor = 0;
	params.bgthreshold = 80;
	params.maxintensitydev = 40;
	params.maxhangle = 0.4; // PI/8
	params.maxvangle = 0.4; // PI/8
	params.minrad = 2;
	params.maxrad = 6;
	params.maxcylrot = 1; // PI/3
	params.winwidth = 24;
	params.winheight = 24;
	params.maxscale = -1.0;
	params.random = false;
}
