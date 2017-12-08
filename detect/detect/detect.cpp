#include "opencv2/opencv.hpp" 
#include "ConfigReader.h"
#include <filesystem>
#include <vector>

#define CONFIG "detect.cfg"

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem;

typedef struct {
	fs::path image;
	Rect area;
	int level;
	double weight;
} detect_stat;

typedef vector<detect_stat> detect_vec;

int detect(const fs::path& image_file, const ConfigReader& config, detect_vec& detected)
{
	string cascade_file;
	int image_resize_width = 0, minNeighbours = 3, minSize = 0, maxSize = 0;
	double scaleFactor = 1.1;

	config.GetParamValue("cascade_file", cascade_file);
	config.GetParamValue("cc_minSize", minSize);
	config.GetParamValue("cc_maxSize", maxSize);
	config.GetParamValue("cc_minNeighbours", minNeighbours);
	config.GetParamValue("cc_scaleFactor", scaleFactor);
	config.GetParamValue("image_resize_width", image_resize_width);

	Mat image_src = imread(image_file.string(), IMREAD_COLOR);
	if (image_src.empty()) throw;

	if (image_resize_width)
	{
		double scale = (double)image_resize_width / image_src.cols;
		resize(image_src, image_src, Size(), scale, scale);
	}

	Mat image_grey;
	cvtColor(image_src, image_grey, COLOR_BGR2GRAY);

	vector<Rect> detected_areas;
	vector<int> reject_levels;
	vector<double> level_weights;
	detect_stat stat;
	CascadeClassifier cascade(cascade_file);
	Size min(minSize, minSize), max(maxSize, maxSize);

	cascade.detectMultiScale(image_grey, detected_areas, reject_levels, level_weights, scaleFactor, minNeighbours, 0, min, max, true);

	int i;
	for (i = 0; i < detected_areas.size(); i++)
	{
		stat.image = image_file;
		stat.area = detected_areas[i];
		stat.level = reject_levels[i];
		stat.weight = level_weights[i];
		detected.push_back(stat);
	}

	return i;
}

int output(const ConfigReader& config, detect_vec& detected, bool verbose = true)
{
	string output_dir;
	string action = "outline";
	int crop_num = 0;

	config.GetParamValue("output_dir", output_dir);
	config.GetParamValue("action", action);
	config.GetParamValue("crop_num", crop_num);

	if (output_dir[output_dir.length() - 1] == '\\') output_dir.erase(output_dir.length() - 1, 1);

	fs::create_directory(output_dir);

	if (action == "crop_min_weight")
	{
		sort(detected.begin(), detected.end(), [](detect_stat a, detect_stat b) {
			return a.weight < b.weight;
		});

		if (crop_num < detected.size())
		{
			detected.erase(detected.begin() + crop_num, detected.end());
		}
	}
	else if (action == "crop_max_weight")
	{
		sort(detected.begin(), detected.end(), [](detect_stat a, detect_stat b) {
			return a.weight > b.weight;
		});

		if (crop_num < detected.size())
		{
			detected.erase(detected.begin() + crop_num, detected.end());
		}
	}

	string curimage;
	Mat image_src;
	int i;
	for (i = 0; i < detected.size(); i++)
	{
		if (detected[i].image.string() != curimage)
		{
			curimage = detected[i].image.string();
			image_src = imread(curimage, IMREAD_COLOR);
			if (image_src.empty()) throw;
		}

		if (action == "outline")
		{
			rectangle(image_src, detected[i].area, Scalar(255, 0, 255), 2, LINE_4);
			char notes[50];
			sprintf_s(notes, "%d, %.2f", detected[i].level, detected[i].weight);
			putText(image_src, notes, Point(detected[i].area.x + 5, detected[i].area.y + 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 255, 255), 1);

			if (i + 1 == detected.size() || detected[i + 1].image.string() != curimage)
			{
				string output_file = output_dir + '\\' + detected[i].image.stem().string() + "_outline" + detected[i].image.extension().string();

				if (verbose)
					printf("Writing %s...\n", output_file.c_str());

				imwrite(output_file, image_src);
			}
		}
		else if (action == "crop" || action == "crop_min_weight" || action == "crop_max_weight")
		{
			Mat crop = image_src(detected[i].area);
			char buf[12];
			sprintf_s(buf, 12, "%04d_%.2f", i, detected[i].weight);
			string output_file = output_dir + '\\' + detected[i].image.stem().string() + "_crop" + buf + detected[i].image.extension().string();

			if (verbose)
				printf("Writing %s...\n", output_file.c_str());

			imwrite(output_file, crop);
		}
	}

	return i;
}

int main(int argc, char* argv[])
{
	ConfigReader config;

	if (!config.Create(CONFIG))
	{
		printf("Usage: %s\n%s config file must exist.\n", argv[0], CONFIG);
		exit(1);
	}

	string input_dir, input_file, input_type;
	config.GetParamValue("input_dir", input_dir);
	config.GetParamValue("input_file", input_file);
	config.GetParamValue("input_type", input_type);

	vector<fs::path> input_files;
	if (input_file.size())
	{
		input_files.push_back(fs::path(input_file));
	}
	else
	{
		fs::directory_iterator begin(input_dir), end;
		copy_if(begin, end, std::back_inserter(input_files), [](const fs::path& path) {
			return fs::is_regular_file(path) && (path.extension() == ".jpg" || path.extension() == ".JPG");
		});
	}

	srand((int)time(NULL));
	detect_vec detected;

	if (input_type == "random")
	{
		int i = 0;
		while (i < input_files.size())
		{
			size_t f = (int)((double)rand() / RAND_MAX * (input_files.size() - 1));
			printf("#%d %s detecting: ", i, input_files[f].string().c_str());
			printf("%d detected\n", detect(input_files[f].string(), config, detected));
			i++;
		}
	}
	else
	{
		vector<fs::path>::iterator it = input_files.begin();
		for (int i = 0; it != input_files.end(); i++, it++)
		{
			printf("#%d %s detecting: ", i, (*it).string().c_str());
			printf("%d detected\n", detect((*it).string(), config, detected));
		}
	}

	int d = output(config, detected);

	return d;
}
