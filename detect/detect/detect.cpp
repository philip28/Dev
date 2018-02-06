#include "opencv2/opencv.hpp" 
#include "ConfigReader.h"
#include <vector>
#include <thread>
#include "svm.h"
#include "svm_wrapper.h"
#include "boost/filesystem.hpp"

#if defined HAVE_TBB
#include "tbb/tbb.h"
#include "tbb/concurrent_vector.h"

using namespace tbb;

typedef spin_mutex vector_mutex_type;
#endif

#define CONFIG "detect.config"

using namespace std;
using namespace cv;

typedef struct
{
	string fullname;
	string name;
	string path;
	string ext;
} file_info;

typedef struct
{
	boost::filesystem::path image_file;
	Rect area;
	int classid;
	int level;
	double weight;
} detect_info;


void svm_hog_detectMultiScale(svm_model* model,
	Mat& image,
	vector<detect_info>& info,
	Size winsize = Size(32,32),
	int stepsize = 4,
	double scale_factor = 1.1,
	int minNeighbors = 3,
	Size minsize = Size(),
	Size maxsize = Size(),
	bool extended_log = true)
{
	Size imgsize = image.size();

	if (maxsize.height == 0 || maxsize.width == 0)
		maxsize = imgsize;

	if (maxsize.width > imgsize.width || maxsize.height > imgsize.height)
		maxsize = imgsize;
	
	if (minsize.width < winsize.width || minsize.height > winsize.height)
		minsize = winsize;

	if (maxsize.width < minsize.width || maxsize.height < minsize.height)
		return;

	if (imgsize.height < minsize.height || imgsize.width < minsize.width)
		return;


	std::vector<double> scales;
	scales.reserve(1024);

	double minfactor = (double)imgsize.height / (double)maxsize.height;
	double maxfactor = (double)imgsize.height / (double)minsize.height;
	double whratio = (double)imgsize.width / (double)imgsize.height;

	for (double factor = minfactor; factor <= maxfactor; factor *= scale_factor)
	{
		scales.push_back(factor);
	}

	size_t nscales = scales.size();
	for (size_t scale_idx = 0; scale_idx < nscales; scale_idx++)
	{
		double scale = scales[scale_idx];
		Mat scaled_image;
		scaled_image.create(cvRound(winsize.height*scale), cvRound(winsize.height*scale*whratio), CV_8UC1);
		resize(image, scaled_image, scaled_image.size());

		int yend = scaled_image.rows - winsize.height;
		int xend = scaled_image.cols - winsize.width;

		if (extended_log)
			printf("%f, %d, %d\n", scale, xend, yend);

#if defined HAVE_TBB
		vector_mutex_type vector_mutex;

		parallel_for(blocked_range2d<int>(0, yend / stepsize + 1, 1, 0, xend / stepsize + 1, 1),
			[&scaled_image, model, &info, &imgsize, &winsize, &stepsize, &vector_mutex](const blocked_range2d<int>& r)
		{
			for (int y = r.rows().begin(); y != r.rows().end(); ++y)
			{
				for (int x = r.cols().begin(); x != r.cols().end(); ++x)
				{
					int xm = x * stepsize, ym = y * stepsize;
					Rect area(xm, ym, winsize.width, winsize.height);

					HOGDescriptor hog;
					hog.winSize = winsize;
					vector<float> descriptors;
					hog.compute(scaled_image(area), descriptors, Size(8, 8), Size(0, 0));

					Mat desc(descriptors);
					struct svm_node* node;
					svm_problem_wrapper::mat_to_svm_node(desc, node);
					double predict_label = svm_predict(model, node);

					if (predict_label > 0)
					{
						double wratio = (double)imgsize.width / (double)scaled_image.cols;
						double hratio = (double)imgsize.height / (double)scaled_image.rows;
						detect_info d;
						d.area = Rect(cvRound(area.x * wratio), cvRound(area.y * hratio),
							cvRound(area.width * wratio), cvRound(area.height * hratio));
						d.classid = (int)predict_label;
						d.weight = 1;
						d.level = 0;
						
						{
							vector_mutex_type::scoped_lock lock(vector_mutex);
							info.push_back(d);
						}
					}

					delete node;

//				cout << "thread: " << std::this_thread::get_id() << ", xm: " << xm << ", ym: " << ym << endl;
				}
			}
		});

#else
		for (int y = 0; y < yend; y += stepsize)
		{
			for (int x = 0; x < xend; x += stepsize)
			{
				Rect area(x, y, winsize.width, winsize.height);
				HOGDescriptor hog;
				hog.winSize = winsize;
				vector<float> descriptors;
				hog.compute(scaled_image(area), descriptors, Size(8, 8), Size(0, 0));

				Mat desc(descriptors);
				struct svm_node* node;
				svm_problem_wrapper::mat_to_svm_node(desc, node);
				double predict_label = svm_predict(model, node);

				if (predict_label > 0)
				{
					double wratio = (double)imgsize.width / (double)scaled_image.cols;
					double hratio = (double)imgsize.height / (double)scaled_image.rows;
					detect_info d;
					d.area = Rect(cvRound(area.x * wratio), cvRound(area.y * hratio),
						cvRound(area.width * wratio), cvRound(area.height * hratio));
					d.classid = (int)predict_label;
					d.weight = 1;
					d.level = 0;
					info.push_back(d);
				}

				delete node;
			}
		}
#endif
	}
}

int detect(const boost::filesystem::path& image_file, const ConfigReader& config, vector<detect_info>& info, bool extended_log = true)
{
	int detect_type;
	string model_file;
	int image_resize_width = 0, minNeighbours = 3, minsize = 0, maxsize = 0;
	int winSizeX = 32, winSizeY = 32;
	double scaleFactor = 1.1;

	config.GetParamValue("detect_type", detect_type);
	config.GetParamValue("model_file", model_file);
	config.GetParamValue("winSizeX", winSizeX);
	config.GetParamValue("winSizeY", winSizeY);
	config.GetParamValue("minSize", minsize);
	config.GetParamValue("maxSize", maxsize);
	config.GetParamValue("minNeighbours", minNeighbours);
	config.GetParamValue("scaleFactor", scaleFactor);
	config.GetParamValue("image_resize_width", image_resize_width);

	Mat image_src = imread(image_file.string(), IMREAD_COLOR);
	if (image_src.empty()) throw;

	if (image_resize_width)
	{
		double scale = (double)image_resize_width / image_src.cols;
		resize(image_src, image_src, Size(), scale, scale);
	}

	if (detect_type == 0)
	{
		vector<Rect> detected_areas;
		vector<int> detected_classes;
		vector<int> reject_levels;
		vector<double> level_weights;
		Size minarea(minsize, minsize), maxarea(maxsize, maxsize);

		Mat image_grey;
		cvtColor(image_src, image_grey, COLOR_BGR2GRAY);
		CascadeClassifier cascade(model_file);
		cascade.detectMultiScale(image_grey, detected_areas, reject_levels, level_weights, scaleFactor, minNeighbours, 0, minarea, maxarea, true);

		for (int i = 0; i < detected_areas.size(); i++)
		{
			detect_info d;
			d.image_file = image_file;
			d.area = detected_areas[i];
			d.weight = level_weights[i];
			d.level = reject_levels[i];
			d.classid = 1;
			info.push_back(d);
		}
	}

	else if (detect_type == 1)
	{
		Mat image_grey;
		cvtColor(image_src, image_grey, COLOR_BGR2GRAY);

		svm_model* model;
		model = svm_load_model(model_file.c_str());

		vector<detect_info> detected;
		Size winsize(winSizeX, winSizeY);
		Size minarea(minsize, minsize), maxarea(maxsize, maxsize);

		svm_hog_detectMultiScale(model, image_grey, detected, winsize, 4, scaleFactor, minNeighbours, minarea, maxarea, extended_log);
		svm_free_and_destroy_model(&model);

		for_each(detected.begin(), detected.end(), [&image_file, &info](detect_info& r)
		{
			r.image_file = image_file;
			info.push_back(r);
		});
	}
	else
		throw;

	return (int)info.size();
}

int output(const ConfigReader& config, vector<detect_info>& info, bool extended_log = true)
{
	Scalar palette[] = { {255, 0, 255}, {0, 255, 255}, {255, 255, 0}, {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {255, 255, 255} };

	string output_dir;
	string action = "outline";
	int crop_num = 0;

	config.GetParamValue("output_dir", output_dir);
	config.GetParamValue("action", action);
	config.GetParamValue("crop_num", crop_num);

	if (output_dir[output_dir.length() - 1] == '\\') output_dir.erase(output_dir.length() - 1, 1);

	boost::filesystem::path p(output_dir);
	if (!boost::filesystem::exists(p))
		boost::filesystem::create_directory(p);

	if (action == "crop_min_weight")
	{
		sort(info.begin(), info.end(), [](detect_info a, detect_info b) {
			return a.weight < b.weight;
		});

		if (crop_num < info.size())
		{
			info.erase(info.begin() + crop_num, info.end());
		}
	}
	else if (action == "crop_max_weight")
	{
		sort(info.begin(), info.end(), [](detect_info a, detect_info b) {
			return a.weight > b.weight;
		});

		if (crop_num < info.size())
		{
			info.erase(info.begin() + crop_num, info.end());
		}
	}

	string curimage;
	Mat image_src;
	int i;
	for (i = 0; i < info.size(); i++)
	{
		if (info[i].image_file.string() != curimage)
		{
			curimage = info[i].image_file.string();
			image_src = imread(curimage, IMREAD_COLOR);
			if (image_src.empty()) throw;
		}

		if (action == "outline")
		{
			rectangle(image_src, info[i].area, palette[info[i].classid], 2, LINE_4);
			char notes[50];
			sprintf_s(notes, "%d: %.2f", info[i].classid, info[i].weight);
			putText(image_src, notes, Point(info[i].area.x + 5, info[i].area.y + 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 255, 255), 1);

			if (i + 1 == info.size() || info[i + 1].image_file.string() != curimage)
			{
				string output_file = output_dir + '\\' + info[i].image_file.stem().string() + "_outline" + info[i].image_file.extension().string();

				if (extended_log)
					printf("Writing %s...\n", output_file.c_str());

				imwrite(output_file, image_src);
			}
		}
		else if (action == "crop" || action == "crop_min_weight" || action == "crop_max_weight")
		{
			Mat crop = image_src(info[i].area);
			char buf[12];
			sprintf_s(buf, 16, "%04d_%d_%.2f", i, info[i].classid, info[i].weight);
			string output_file = output_dir + '\\' + info[i].image_file.stem().string() + "_crop" + buf + info[i].image_file.extension().string();

			if (extended_log)
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
	bool extended_log;
	config.GetParamValue("input_dir", input_dir);
	config.GetParamValue("input_file", input_file);
	config.GetParamValue("input_type", input_type);
	config.GetParamValue("extended_log", extended_log);

	vector<boost::filesystem::path> input_files;
	if (input_file.size())
		input_files.push_back(boost::filesystem::path(input_file));
	else
		for (auto&& x : boost::filesystem::directory_iterator(input_dir))
			input_files.push_back(x.path());

	srand((int)time(NULL));

	vector<detect_info> detected;
	if (input_type == "random")
	{
		int i = 0;
		while (i < input_files.size())
		{
			size_t f = (int)((double)rand() / RAND_MAX * (input_files.size() - 1));
			printf("#%d %s detecting... ", i, input_files[f].string().c_str());
			printf("...[done] %d detected\n", detect(input_files[f], config, detected, extended_log));
			i++;
		}
	}
	else
	{
		int i = 0;
		while (i < input_files.size())
		{
			printf("#%d %s detecting... ", i, input_files[i].filename().string().c_str());
			printf("...[done] %d detected\n", detect(input_files[i], config, detected, extended_log));
			i++;
		}
	}

	printf("Writing output...");
	int d = output(config, detected);
	printf("...[done]\n");

	return d;
}
