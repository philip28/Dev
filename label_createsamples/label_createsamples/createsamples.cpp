#include "utility.h"
#include "ConfigReader.h"
#include "ImageTransformData.h"

using namespace std;

int main(int argc, char* argv[])
{
	string imagename;
	string config_file = "label_createsamples.config";
	ImageTransformData data;
	ConfigReader config;
	bool visualize = false;

	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "-visualize"))
			visualize = true;
		else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "-config"))
			config_file = argv[i + 1];
	}

	if (!config.Create(config_file))
	{
		printf("Usage: %s [-c | config]\n", argv[0]);
		exit(1);
	}

	config.GetParamValue("image_file", imagename);
	config.GetParamValue("output_dir", data.output_dir);
	config.GetParamValue("info_file", data.info_file);
	config.GetParamValue("background_dir", data.background_dir);
	config.GetParamValue("bgfile", data.bgname);
	config.GetParamValue("num_bg_files", data.params.num_bg_files);
	config.GetParamValue("num_samples_per_file", data.params.num_samples_per_file);
	config.GetParamValue("output_type", data.params.output_type);
	config.GetParamValue("randomize", data.params.randomize);
	config.GetParamValue("grayscale", data.params.grayscale);
	config.GetParamValue("bg_color", data.params.bg_color);
	config.GetParamValue("fill_color", data.params.fill_color);
	config.GetParamValue("transparent_color_low", data.params.transparent_color_low);
	config.GetParamValue("transparent_color_high", data.params.transparent_color_high);
	config.GetParamValue("noise_removal", data.params.noise_removal);
	config.GetParamValue("maxintencitydev", data.params.maxintensitydev);
	config.GetParamValue("maxxrotangle", data.params.maxxangle);
	config.GetParamValue("maxyrotangle", data.params.maxyangle);
	config.GetParamValue("maxzrotangle", data.params.maxzangle);
	config.GetParamValue("mincylrad", data.params.minrad);
	config.GetParamValue("maxcylrad", data.params.maxrad);
	config.GetParamValue("maxcylincl", data.params.maxincl);
	config.GetParamValue("winsize", data.params.winsize);
	config.GetParamValue("fixed_size", data.params.fixed_size);
	config.GetParamValue("maxscale", data.params.maxscale);
	config.GetParamValue("random_bg_file", data.params.random_bg_file);
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

	if (visualize)
	{
		Visualize(imagename, &data);
	}
	else
	{
		printf("Info file name: %s\n", data.info_file.c_str());
		printf("Img file name: %s\n", imagename.c_str());
		printf("BG  file name: %s\n", data.bgname.c_str());
		printf("Num: %d\n", data.params.num_bg_files);
		printf("BG color: %s\n", data.params.fill_color.c_str());
		printf("Max intensity deviation: %d\n", data.params.maxintensitydev);
		printf("Max x angle: %g rad\n", data.params.maxxangle);
		printf("Max y angle: %g rad\n", data.params.maxyangle);
		printf("Max z angle: %g rad\n", data.params.maxzangle);
		printf("Min wrapping radius: %g\n", data.params.minrad);
		printf("Max wrapping radius: %g\n", data.params.maxrad);
		printf("Max inclination: %g\n", data.params.maxincl);
		printf("Window size: %d\n", data.params.winsize);
		printf("Max Scale: %g\n", data.params.maxscale);
		if (data.params.random_bg_file) printf("Using random background file selection mode\n");

		printf("Creating training samples from single image applying distortion+shading...\n");

		if (data.params.randomize)
			srand((int)time(NULL));

		CreateTestSamples(imagename, &data);
	}

	return 0;
}
