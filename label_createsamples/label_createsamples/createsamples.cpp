#include "utility.h"
#include "ConfigReader.h"
#include "ImageTransformData.h"

#define CONFIG "label_createsamples.cfg"

int main(int argc, char* argv[])
{
	string infoname;
	string imagename;
	string bgname;
	ImageTransformData data;
	ConfigReader config;
	bool visualize = false;

	if (!config.Create(CONFIG))
	{
		printf("Usage: %s\n%s config file must exist.\n", argv[0], CONFIG);
		exit(1);
	}

	config.GetParamValue("infofile", infoname);
	config.GetParamValue("imagefile", imagename);
	config.GetParamValue("bgfile", bgname);
	config.GetParamValue("num", data.params.numsample);
	config.GetParamValue("bg_fill_color", data.params.bg_fill_color);
	config.GetParamValue("transparency_low", data.params.transparent_color_low);
	config.GetParamValue("transparency_high", data.params.transparent_color_high);
	config.GetParamValue("noise_removal", data.params.noise_removal);
	config.GetParamValue("maxintencitydev", data.params.maxintensitydev);
	config.GetParamValue("maxxrotangle", data.params.maxxangle);
	config.GetParamValue("maxyrotangle", data.params.maxyangle);
	config.GetParamValue("maxzrotangle", data.params.maxzangle);
	config.GetParamValue("mincylrad", data.params.minrad);
	config.GetParamValue("maxcylrad", data.params.maxrad);
	config.GetParamValue("maxcylincl", data.params.maxincl);
	config.GetParamValue("winsize", data.params.winsize);
	config.GetParamValue("maxscale", data.params.maxscale);
	config.GetParamValue("random", data.params.random);
	config.GetParamValue("albedo_min", data.params.albedo_min);
	config.GetParamValue("albedo_max", data.params.albedo_max);
	config.GetParamValue("ambient", data.params.ambient);
	config.GetParamValue("specular_min", data.params.specular_min);
	config.GetParamValue("specular_max", data.params.specular_max);
	config.GetParamValue("shininess_min", data.params.shininess_min);
	config.GetParamValue("shininess_max", data.params.shininess_max);
	config.GetParamValue("light_color", data.params.light_color);
	config.GetParamValue("light_intensity_min", data.params.light_intensity_min);
	config.GetParamValue("light_intensity_max", data.params.light_intensity_max);
	config.GetParamValue("light_dir_dev_max", data.params.light_dir_dev_max);

	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-visualize"))
			visualize = true;

	}

	if (visualize)
	{
		Visualize(imagename, &data);
	}
	else
	{
		printf("Info file name: %s\n", infoname.c_str());
		printf("Img file name: %s\n", imagename.c_str());
		printf("BG  file name: %s\n", bgname.c_str());
		printf("Num: %d\n", data.params.numsample);
		printf("BG color: %s\n", data.params.bg_fill_color.c_str());
		printf("Max intensity deviation: %d\n", data.params.maxintensitydev);
		printf("Max x angle: %g rad\n", data.params.maxxangle);
		printf("Max y angle: %g rad\n", data.params.maxyangle);
		printf("Max z angle: %g rad\n", data.params.maxzangle);
		printf("Min wrapping radius: %g\n", data.params.minrad);
		printf("Max wrapping radius: %g\n", data.params.maxrad);
		printf("Max inclination: %g\n", data.params.maxincl);
		printf("Window size: %d\n", data.params.winsize);
		printf("Max Scale: %g\n", data.params.maxscale);
		if (data.params.random) printf("Using random mode\n");

		printf("Creating training samples from single image applying distortion+shading...\n");

		CreateTestSamples(infoname, imagename, bgname, &data);
	}

	return 0;
}
