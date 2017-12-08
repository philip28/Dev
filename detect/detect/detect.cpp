#include "opencv2/opencv.hpp" 
#include "ConfigReader.h"
#include <filesystem>
#include <vector>

#define CONFIG "detect.cfg"

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem;

int detect(const fs::path& image_file, const ConfigReader& config)
{
	string cascade_file, output_dir;
	string action = "outline";
	int image_resize_width = 0, minNeighbours = 3, minSize = 0, maxSize = 0;
	double scaleFactor = 1.1;

	config.GetParamValue("cascade_file", cascade_file);
	config.GetParamValue("cc_minSize", minSize);
	config.GetParamValue("cc_maxSize", maxSize);
	config.GetParamValue("cc_minNeighbours", minNeighbours);
	config.GetParamValue("cc_scaleFactor", scaleFactor);
	config.GetParamValue("image_resize_width", image_resize_width);
	config.GetParamValue("output_dir", output_dir);
	config.GetParamValue("action", action);

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
	CascadeClassifier cascade(cascade_file);
	Size min(minSize, minSize), max(maxSize, maxSize);

	cascade.detectMultiScale(image_grey, detected_areas, scaleFactor, minNeighbours, 0, min, max);

	if (output_dir[output_dir.length() - 1] == '\\') output_dir.erase(output_dir.length() - 1, 1);

	if (action == "outline")
	{
		for (int i = 0; i < detected_areas.size(); i++)
		{
			rectangle(image_src, detected_areas[i], Scalar(255, 0, 255), 2, LINE_4);
		}
		if (detected_areas.size())
		{
			string output_file = output_dir + '\\' + image_file.stem().string() + "_outline" + image_file.extension().string();
			imwrite(output_file, image_src);
		}
	}
	else if (action == "crop")
	{
		for (int i = 0; i < detected_areas.size(); i++)
		{
			Mat crop = image_src(detected_areas[i]);
			char buf[4];
			sprintf_s(buf, 4, "%03d", i);
			string output_file = output_dir + '\\' + image_file.stem().string() + "_crop" + buf + image_file.extension().string();
			imwrite(output_file, crop);
		}
	}

	return (int)detected_areas.size();
}

int main(int argc, char* argv[])
{
	ConfigReader config;

	if (!config.Create(CONFIG))
	{
		printf("Usage: %s\n%s config file must exist.\n", argv[0], CONFIG);
		exit(1);
	}

	string input_dir, input_file;
	config.GetParamValue("input_dir", input_dir);
	config.GetParamValue("input_file", input_file);

	vector<fs::path> input_files;
	if (input_file.size())
	{
		input_files.push_back(fs::path(input_file));
	}
	else
	{
		fs::directory_iterator begin(input_dir), end;
		copy_if(begin, end, std::back_inserter(input_files), [](const fs::path& path) {
			return fs::is_regular_file(path) && (path.extension() == ".jpg");
		});
	}

	vector<fs::path>::iterator it = input_files.begin();
	for (int i = 0; it != input_files.end(); i++, it++)
	{
		printf("#%d %s: %d detected\n", i, (*it).string().c_str(), detect((*it).string(), config));
	}

	return 0;
}
