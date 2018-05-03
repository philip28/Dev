#include "SampleMaker.h"
#include "ConfigReader.h"
#include "RandGen.h"

using namespace std;

int main(int argc, char* argv[])
{
	string config_file = "label_createsamples.config";
	param_type params;
	SampleMaker sample_maker;
	ConfigReader config;
	bool visualize = false;

	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "-visualize"))
			visualize = true;
		else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "-config"))
			config_file = argv[i + 1];
	}

	if (!config.Init(config_file))
	{
		printf("Usage: %s [-c | config]\n", argv[0]);
		exit(1);
	}

	config.InitParam("image_dir", "Input images directory", params.image_dir);
	config.InitParam("image_file", "Single input image", params.image_file);
	config.InitParam("output_dir", "Output samples directory", params.output_dir);
	config.InitParam("output_dir_bg", "Output background directory (only for +patch mode)", params.output_dir_bg);
	config.InitParam("info_file", "Generate info file", params.info_file);
	config.InitParam("background_dir", "Input background images directory", params.background_dir);
	config.InitParam("background_file", "Single input background image", params.background_file);
	config.InitParam("background_list", "Text file with input background image list", params.background_list);
	config.InitParam("num_samples", "Number of samples to create", params.num_samples);
	config.InitParam("num_patches", "Number of patches generated per background file", params.num_patches);
	config.InitParam("output_type", "Sample output type", params.output_type);
	config.InitParam("randomize", "Initialize random number generator", params.randomize);
	config.InitParam("grayscale", "Use grayscale image processing", params.grayscale);
	config.InitParam("bg_color", "Background color", params.bg_color);
	config.InitParam("fill_color", "Fill color", params.fill_color);
	config.InitParam("transparent_color_low", "Lower threshold for imput image transparency", params.transparent_color_low);
	config.InitParam("transparent_color_high", "Upper threshold for imput image transparency", params.transparent_color_high);
	config.InitParam("noise_removal", "Apply noise removal after transformations", params.noise_removal);
	config.InitParam("transformation", "Type of transformation", params.transformation);
	config.InitParam("maxxrotangle", "Max X rotation angle (radians)", params.maxxangle);
	config.InitParam("maxyrotangle", "Max Y rotation angle (radians)", params.maxyangle);
	config.InitParam("maxzrotangle", "Max Z rotation angle (radians)", params.maxzangle);
	config.InitParam("minradius", "Min cylinder radius (multiple of input image width / PI)", params.minrad);
	config.InitParam("maxradius", "Max cylinder radius (multiple of input image width / PI)", params.maxrad);
	config.InitParam("maxincl", "Max cyliner inclination (radians)", params.maxincl);
	config.InitParam("winsize", "Patch scaling granularity", params.winsize);
	config.InitParam("patch_size", "Use fixed square patch size", params.patch_size);
	config.InitParam("min_image_scale", "Vary object size (from minimum to full scale)", params.min_image_scale);
	config.InitParam("vary_image_pos", "Vary object position within sample", params.vary_image_pos);
	config.InitParam("random_bg_file", "Use random background file sequence", params.random_bg_file);
	config.InitParam("use_lighting", "Apply lighting", params.use_lighting);
	config.InitParam("albedo_min", "Lower threshold for surface albedo", params.albedo_min);
	config.InitParam("albedo_max", "Upper threshold for surface albedo", params.albedo_max);
	config.InitParam("ambient", "Weight of the ambient surface component", params.ambient);
	config.InitParam("specular_min", "Lower threshold for specular surface component", params.specular_min);
	config.InitParam("specular_max", "Upper threshold for specular surface component", params.specular_max);
	config.InitParam("shininess_min", "Lower threshold for surface shineness", params.shininess_min);
	config.InitParam("shininess_max", "Upper threshold for surface shineness", params.shininess_max);
	config.InitParam("light_temp_min", "Lower threshold for light temperature (K)", params.light_temp_min);
	config.InitParam("light_temp_max", "Upper threshold for light temperature (K)", params.light_temp_max);
	config.InitParam("light_bias_max", "Max bias applied to light color", params.light_bias_max);
	config.InitParam("hue_bias_max", "Max bias applied to Hue HSV component", params.hue_bias_max);
	config.InitParam("saturation_bias_max", "Max bias applied to Saturation HSV component", params.saturation_bias_max);
	config.InitParam("value_bias_max", "Max bias applied to Value HSV component", params.value_bias_max);
	config.InitParam("light_intensity_min", "Lower threshold for light intensity", params.light_intensity_min);
	config.InitParam("light_intensity_max", "Upper threshold for light intensity", params.light_intensity_max);
	config.InitParam("light_dir_dev_max", "Max deviation from reflection field of view (radians)", params.light_dir_dev_max);

	if (visualize)
	{
		SampleMaker sm;
		sm.params = params;
		sm.Visualize(params.image_file);
	}
	else
	{
		config.Print();
		printf("Creating training samples from single image applying distortion+shading...\n");

		unsigned int seed = 0;
		if (params.randomize)
			seed = (unsigned int)time(0);
		RandGen random(seed);

		ImageReader imreader, bgreader;
		InfoWriter infowriter;

		if (!params.image_file.empty()) {
			if (!imreader.Init(params.image_file, input_type::IMAGE, &random, params.grayscale))
				CV_Error(CV_StsBadArg, "error reading source image\n");
		}
		else if (!params.image_dir.empty()) {
			if (!imreader.Init(params.image_dir, input_type::DIR, &random, params.grayscale))
				CV_Error(CV_StsBadArg, "error reading source image list\n");
		}
		else
			CV_Error(CV_StsBadArg, "source image is missing\n");

		if (params.output_dir.empty())
			CV_Error(CV_StsBadArg, "output dir is missing\n");

		if (params.output_type != "nobg")
		{
			if (!params.background_file.empty()) {
				if (!bgreader.Init(params.background_file, input_type::IMAGE, &random, params.grayscale))
					CV_Error(CV_StsBadArg, "error reading background image\n");
			}
			else if (!params.background_dir.empty()) {
				if (!bgreader.Init(params.background_dir, input_type::DIR, &random, params.grayscale))
					CV_Error(CV_StsBadArg, "error reading background image list\n");
			}
			else if (!params.background_list.empty()) {
				if (!bgreader.Init(params.background_list, input_type::LIST, &random, params.grayscale))
					CV_Error(CV_StsBadArg, "error reading background list\n");
			}
			else
				CV_Error(CV_StsBadArg, "background image is missing\n");
		}
		else
		{
			printf("Proceeding without background\n");
			bgreader.rec_count = params.num_samples;
			params.num_patches = 1;
		}


		if (!infowriter.Init(params.output_dir, params.info_file, params.output_dir_bg, &random))
			CV_Error(CV_StsBadArg, "error initializing output files\n");

		if (params.num_samples > 0 && params.num_samples < bgreader.rec_count)
			bgreader.rec_count = params.num_samples;

		while (imreader.Next() != -1)
		{
#if defined HAVE_TBB
			tbb::spin_mutex mutex;
			int index = 0;

			tbb::parallel_for(tbb::blocked_range<size_t>(0, bgreader.rec_count, 1),
				[&](const tbb::blocked_range<size_t>& r)
			{
				SampleMaker sm;
				sm.params = params;
				sm.imreader = imreader;
				sm.bgreader = bgreader;
				sm.infowriter = &infowriter;
				sm.random = &random;

				for (auto rec = r.begin(); rec != r.end(); ++rec)
				{
					if (params.random_bg_file)
						sm.bgreader.GetRandom();
					else
						sm.bgreader.Get(rec);

					if (params.output_type != "nobg")
					{
						tbb::spin_mutex::scoped_lock lock(mutex);
						index++;
						std::cout << "Processing background image #" << index << ": " << sm.bgreader.imagefullname << std::endl;
					}
					else
					{
						tbb::spin_mutex::scoped_lock lock(mutex);
						std::cout << "Processing sample #" << sm.bgreader.rec_index << std::endl;
					}

					sm.ProcessSample();
				}
			});

#else
			int index = 0;
			SampleMaker sm;
			sm.params = params;
			sm.imreader = imreader;
			sm.bgreader = bgreader;
			sm.infowriter = &infowriter;
			sm.random = &random;

			bgreader.Rewind();
			while (sm.bgreader.Next(params.random_bg_file) != -1)
			{
				if (params.output_type != "nobg")
					std::cout << "Processing background image #" << ++index << ": " << sm.bgreader.imagefullname << std::endl;
				else
					std::cout << "Processing sample #" << sm.bgreader.rec_index << std::endl;

				sm.ProcessSample();
			}
#endif
		}
	}

	return 0;
}
