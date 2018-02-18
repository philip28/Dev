#include "opencv2/opencv.hpp" 
#include "ConfigReader.h"
#include <vector>
#include <thread>
#include <iostream>
#include "svm.h"
#include "svm_wrapper.h"
#include "boost/filesystem.hpp"

#if defined HAVE_TBB
#include "tbb/tbb.h"
#include "tbb/concurrent_vector.h"

using namespace tbb;

typedef spin_mutex vector_mutex_type;
#endif

using namespace std;
using namespace cv;

typedef struct
{
	string input_dir;
	string input_file;
	string input_type;
	string output_dir;
	bool extended_log = false;
	int detect_type;
	string model_file;
	int image_resize_width = 0;
	int minNeighbours = 3;
	int minsize = 0, maxsize = 0;
	int winSizeX = 32, winSizeY = 32;
	double scaleFactor = 1.1;
	string action = "outline";
	int crop_num = 0;
	bool crop_resize = false;
} param_info;



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

int detect(const boost::filesystem::path& image_file, param_info& params, vector<detect_info>& info, bool extended_log = true)
{
	Mat image_src = imread(image_file.string(), IMREAD_COLOR);
	if (image_src.empty()) throw;

	if (params.image_resize_width)
	{
		double scale = (double)params.image_resize_width / image_src.cols;
		resize(image_src, image_src, Size(), scale, scale);
	}

	if (params.detect_type == 0)
	{
		vector<Rect> detected_areas;
		vector<int> detected_classes;
		vector<int> reject_levels;
		vector<double> level_weights;
		Size minarea(params.minsize, params.minsize), maxarea(params.maxsize, params.maxsize);

		Mat image_grey;
		cvtColor(image_src, image_grey, COLOR_BGR2GRAY);
		CascadeClassifier cascade(params.model_file);
		cascade.detectMultiScale(image_grey, detected_areas, reject_levels, level_weights, params.scaleFactor, params.minNeighbours, 0, minarea, maxarea, true);

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

	else if (params.detect_type == 1)
	{
		Mat image_grey;
		cvtColor(image_src, image_grey, COLOR_BGR2GRAY);

		svm_model* model;
		model = svm_load_model(params.model_file.c_str());

		vector<detect_info> detected;
		Size winsize(params.winSizeX, params.winSizeY);
		Size minarea(params.minsize, params.minsize), maxarea(params.maxsize, params.maxsize);

		svm_hog_detectMultiScale(model, image_grey, detected, winsize, 4, params.scaleFactor, params.minNeighbours, minarea, maxarea, extended_log);
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

int output(param_info& params, vector<detect_info>& info, bool extended_log = true)
{
	Scalar palette[] = { {255, 0, 255}, {0, 255, 255}, {255, 255, 0}, {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {255, 255, 255} };

	if (params.output_dir[params.output_dir.length() - 1] == '\\') params.output_dir.erase(params.output_dir.length() - 1, 1);

	boost::filesystem::path p(params.output_dir);
	if (!boost::filesystem::exists(p))
	{
		if (extended_log)
			printf("%s does not exist, creating...\n", p.string().c_str());
		boost::filesystem::create_directories(p);
	}

	if (params.action == "crop_min_weight")
	{
		sort(info.begin(), info.end(), [](detect_info a, detect_info b) {
			return a.weight < b.weight;
		});
	}
	else if (params.action == "crop_max_weight")
	{
		sort(info.begin(), info.end(), [](detect_info a, detect_info b) {
			return a.weight > b.weight;
		});
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

		if (params.action == "outline")
		{
			rectangle(image_src, info[i].area, palette[info[i].classid], 2, LINE_4);
			char notes[50];
			sprintf_s(notes, "%d: %.2f", info[i].classid, info[i].weight);
			putText(image_src, notes, Point(info[i].area.x + 5, info[i].area.y + 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 255, 255), 1);

			if (i + 1 == info.size() || info[i + 1].image_file.string() != curimage)
			{
				string output_file = params.output_dir + '\\' + info[i].image_file.stem().string() + "_outline" + info[i].image_file.extension().string();

				if (extended_log)
					printf("Writing %s...\n", output_file.c_str());

				imwrite(output_file, image_src);
			}
		}
		else if (params.action == "crop" || params.action == "crop_min_weight" || params.action == "crop_max_weight")
		{
			if (params.crop_num > 0 && params.crop_num < info.size())
			{
				info.erase(info.begin() + params.crop_num, info.end());
			}

			Mat crop = image_src(info[i].area);
			if (params.crop_resize)
				resize(crop, crop, Size(params.winSizeX, params.winSizeY));

			char buf[12];
			sprintf_s(buf, 16, "%04d_%d_%.2f", i, info[i].classid, info[i].weight);
			string output_file = params.output_dir + '\\' + info[i].image_file.stem().string() + "_crop" + buf + info[i].image_file.extension().string();

			if (extended_log)
				printf("Writing %s...\n", output_file.c_str());

			imwrite(output_file, crop);
		}
	}

	return i;
}

void get_params(const string& config_file, param_info& params)
{
	ConfigReader config;
	if (!config.Create(config_file))
	{
		printf("Usage: detect [-c | config]\n");
		exit(1);
	}

	config.GetParamValue("input_dir", params.input_dir);
	config.GetParamValue("input_file", params.input_file);
	config.GetParamValue("input_type", params.input_type);
	config.GetParamValue("extended_log", params.extended_log);
	config.GetParamValue("detect_type", params.detect_type);
	config.GetParamValue("model_file", params.model_file);
	config.GetParamValue("winSizeX", params.winSizeX);
	config.GetParamValue("winSizeY", params.winSizeY);
	config.GetParamValue("minSize", params.minsize);
	config.GetParamValue("maxSize", params.maxsize);
	config.GetParamValue("minNeighbours", params.minNeighbours);
	config.GetParamValue("scaleFactor", params.scaleFactor);
	config.GetParamValue("image_resize_width", params.image_resize_width);
	config.GetParamValue("output_dir", params.output_dir);
	config.GetParamValue("action", params.action);
	config.GetParamValue("crop_num", params.crop_num);
	config.GetParamValue("crop_resize", params.crop_resize);
}

int main(int argc, char* argv[])
{
	string config_file = "detect.config";
	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "-config"))
			config_file = argv[i + 1];
	}
	
	param_info params;
	get_params(config_file, params);

	vector<boost::filesystem::path> input_files;
	if (params.input_file.size())
		input_files.push_back(boost::filesystem::path(params.input_file));
	else
		for (auto&& x : boost::filesystem::directory_iterator(params.input_dir))
			input_files.push_back(x.path());

	srand((int)time(NULL));

	vector<detect_info> detected;
	if (params.input_type == "random")
	{
		int i = 0;
		while (i < input_files.size())
		{
			size_t f = (int)((double)rand() / RAND_MAX * (input_files.size() - 1));
			printf("#%d %s detecting... ", i, input_files[f].string().c_str());
			printf("...[done] %d detected\n", detect(input_files[f], params, detected, params.extended_log));
			i++;
		}
	}
	else
	{
		int i = 0;
		while (i < input_files.size())
		{
			printf("#%d %s detecting... ", i, input_files[i].filename().string().c_str());
			printf("...[done] %d detected\n", detect(input_files[i], params, detected, params.extended_log));
			i++;
		}
	}

	printf("Writing output...");
	int d = output(params, detected);
	printf("...[done]\n");

	return d;
}
