#include "utility.h"
#include "ImageTransformData.h"

int main(int argc, char* argv[])
{
	string infoname;
	string imagename;
	string bgname;
	ImageTransformData data;

	if (argc == 1)
	{
		printf("Usage: %s\n  [-info <collection_file_name>]\n"
			"  [-img <image_file_name>]\n"
			"  [-bg <background_file_name>]\n"
			"  [-num <number_of_samples = %d>]\n"
			"  [-bgcolor <background_color = %d>]\n"
			"  [-bgthresh <background_color_threshold = %d>]\n"
			"  [-maxidev <max_intensity_deviation = %d>]\n"
			"  [-maxhangle <max_horiz_rotation_angle = %f>]\n"
			"  [-maxvangle <max_vert_rotation_angle = %f>]\n"
			"  [-minrad <min_wrapping_radius = %f>]\n"
			"  [-maxrad <max_wrapping_radius = %f>]\n"
			"  [-maxcylrot <max_cylinder_rotation = %f>]\n"
			"  [-w <sample_width = %d>]\n  [-h <sample_height = %d>]\n"
			"  [-maxscale <max sample scale = %f>]\n"
			"  [-random = %s]\n",
			argv[0], data.params.numsample, data.params.bgcolor, data.params.bgthreshold, data.params.maxintensitydev,
			data.params.maxhangle, data.params.maxvangle, data.params.minrad, data.params.maxrad, data.params.maxcylrot,
			data.params.winwidth, data.params.winheight, data.params.maxscale, data.params.random ? "true" : "false");

		return 0;
	}

	for (int i = 1; i < argc; ++i)
	{
		if (!_stricmp(argv[i], "-info"))
		{
			infoname = argv[++i];
		}
		else if (!_stricmp(argv[i], "-img"))
		{
			imagename = argv[++i];
		}
		else if (!_stricmp(argv[i], "-bg"))
		{
			bgname = argv[++i];
		}
		else if (!_stricmp(argv[i], "-num"))
		{
			data.params.numsample = atoi(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-bgcolor"))
		{
			data.params.bgcolor = atoi(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-bgthresh"))
		{
			data.params.bgthreshold = atoi(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-maxidev"))
		{
			data.params.maxintensitydev = atoi(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-maxhangle"))
		{
			data.params.maxhangle = atof(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-maxvangle"))
		{
			data.params.maxvangle = atof(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-minrad"))
		{
			data.params.minrad = atof(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-maxrad"))
		{
			data.params.maxrad = atof(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-maxcylrot"))
		{
			data.params.maxcylrot = atof(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-w"))
		{
			data.params.winwidth = atoi(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-h"))
		{
			data.params.winheight = atoi(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-maxscale"))
		{
			data.params.maxscale = atof(argv[++i]);
		}
		else if (!_stricmp(argv[i], "-random"))
		{
			data.params.random = true;
		}
	}

	printf("Info file name: %s\n", infoname.c_str());
	printf("Img file name: %s\n", imagename.c_str());
	printf("BG  file name: %s\n", bgname.c_str());
	printf("Num: %d\n", data.params.numsample);
	printf("BG color: %d\n", data.params.bgcolor);
	printf("BG threshold: %d\n", data.params.bgthreshold);
	printf("Max intensity deviation: %d\n", data.params.maxintensitydev);
	printf("Max Horiz angle: %g rad\n", data.params.maxhangle);
	printf("Max Vert angle: %g rad\n", data.params.maxvangle);
	printf("Min wrapping radius: %g\n", data.params.minrad);
	printf("Max wrapping radius: %g\n", data.params.maxrad);
	printf("Max cylinder rotation: %g rad\n", data.params.maxcylrot);
	printf("Base width: %d\n", data.params.winwidth);
	printf("Base height: %d\n", data.params.winheight);
	printf("Max Scale: %g\n", data.params.maxscale);
	if (data.params.random) printf("Using random mode\n");

	printf("Creating training samples from single image applying distortions...\n");

	CreateTestSamples(infoname, imagename, bgname, &data);

	return 0;
}
