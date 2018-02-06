#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include "ConfigReader.h"
#include "svm.h"
#include "svm_wrapper.h"

using namespace cv;
using namespace std;

#define CONFIG "svm_train.config"

void load_images(const String& dirname, vector<Mat>& img_list, Size patchsize = Size(), int numpatch = 1)
{
	vector<String> files;
	glob(dirname, files);

	for (size_t i = 0; i < files.size(); ++i)
	{
		Mat img = imread(files[i]); 
		if (img.empty())            
		{
			cout << files[i] << " is invalid!" << endl;
			continue;
		}

		if (patchsize.width > 0)
		{
			Rect box;
			box.width = patchsize.width;
			box.height = patchsize.height;

			if (img.cols > box.width && img.rows > box.height)
			{
				for (int i = 0; i < numpatch; i++)
				{
					box.x = rand() % (img.cols - box.width);
					box.y = rand() % (img.rows - box.height);
					Mat patch = img(box);
					img_list.push_back(patch.clone());
				}
			}
		}
		else
			img_list.push_back(img);
	}
}

void compute_HOG(const Size wsize, const vector<Mat> & img_list, vector<Mat> & hog_list)
{
	HOGDescriptor hog;
	hog.winSize = wsize;
	Mat gray;
	vector<float> descriptors;

	for (size_t i = 0; i < img_list.size(); i++)
	{
		if (img_list[i].cols >= wsize.width && img_list[i].rows >= wsize.height)
		{
			Rect r = Rect((img_list[i].cols - wsize.width) / 2,
				(img_list[i].rows - wsize.height) / 2,
				wsize.width,
				wsize.height);
			cvtColor(img_list[i](r), gray, COLOR_BGR2GRAY); // should try calculating HOG for each channel and them picking the one with highest norm
			hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
			hog_list.push_back(Mat(descriptors).clone());
		}
	}
}


int main(int argc, char** argv)
{
	string config_file;
	string model_file_name, neg_dir;
	ConfigReader config;
	int detector_width = 0, detector_height = 0, num_samples = 1;
	bool neg_sampling;

	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "-config"))
			config_file = argv[i + 1];
	}

	if (config_file.empty()) config_file = CONFIG;

	if (!config.Create(CONFIG))
	{
		clog << "Usage: " << argv[0] << " [-c config]" << endl;
		exit(1);
	}

	vector<class_info> class_list;

	for (int i = 1; ; i++)
	{
		class_info c;
		c.cid = i;
		config.GetParamValue("class" + to_string(i), c.path);
		if (!c.path.empty())
			class_list.push_back(c);
		else
			break;
	}

	config.GetParamValue("model_file_name", model_file_name);
	config.GetParamValue("neg_dir", neg_dir);
	config.GetParamValue("detector_width", detector_width);
	config.GetParamValue("detector_height", detector_height);
	config.GetParamValue("neg_sampling", neg_sampling);
	config.GetParamValue("num_samples", num_samples);

	struct svm_parameter param;
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	config.GetParamValue("svm_type", param.svm_type);
	config.GetParamValue("kernel_type", param.kernel_type);
	config.GetParamValue("degree", param.degree);
	config.GetParamValue("gamma", param.gamma);
	config.GetParamValue("coef0", param.coef0);
	config.GetParamValue("cost", param.C);
	config.GetParamValue("nu", param.nu);
	config.GetParamValue("loss_epsilon", param.p);
	config.GetParamValue("cache_size", param.cache_size);
	config.GetParamValue("term_epsilon", param.eps);
	config.GetParamValue("shrinking", param.shrinking);
	config.GetParamValue("probability_estimates", param.probability);
//	config.GetParamValue("weight", param.weight);

	for (int i = 0; i < class_list.size(); i++)
	{
		cout << "Class " << class_list[i].cid << ": Positive images are being loaded...";
		load_images(class_list[i].path, class_list[i].image_list);
		if (class_list[i].image_list.size())
		{
			cout << "...[done]" << endl;
		}
		else
		{
			clog << "no images loaded from " << class_list[i].path << endl;
			exit(1);
		}
	}

	Size pos_image_size = class_list[0].image_list[0].size();
	if (detector_width && detector_height)
	{
		pos_image_size = Size(detector_width, detector_height);
	}
	else
	{
		for (size_t i = 0; i < class_list.size(); ++i)
			for (size_t j = 0; j < class_list[i].image_list.size(); ++j)
				if (class_list[i].image_list[j].size() != pos_image_size)
				{
					clog << "All positive images should be same size!" << endl;
					exit(1);
				}
		pos_image_size = pos_image_size / 8 * 8;
	}

	vector<Mat> neg_list;
	cout << "Negative images are being loaded...";
	if (neg_sampling)
		load_images(neg_dir, neg_list, pos_image_size, num_samples);
	else
		load_images(neg_dir, neg_list);

	if (neg_list.size())
	{
		cout << "...[done]" << endl;
	}
	else
	{
		clog << "no images loaded from " << neg_dir << endl;
		exit(1);
	}

	for (int i = 0; i < class_list.size(); i++)
	{
		cout << "Class " << class_list[i].cid << ": HOG are being calculated...";
		compute_HOG(pos_image_size, class_list[i].image_list, class_list[i].grad_list);
		size_t positive_count = class_list[i].grad_list.size();
		cout << "...[done] (positive count : " << positive_count << ")" << endl;
	}

	vector<Mat> neg_gradient;
	cout << "Histogram of Gradients are being calculated for negative images...";
	compute_HOG(pos_image_size, neg_list, neg_gradient);
	size_t negative_count = neg_gradient.size();
	cout << "...[done] (negative count : " << negative_count << ")" << endl;


	cout << "Training SVM..." << endl;
	svm_problem_wrapper p;
	p.make_mult_class(class_list, neg_gradient);

	const char* error_msg = svm_check_parameter(&p.problem, &param);
	if (error_msg)
	{
		clog << "ERROR: " << error_msg << endl;
		exit(1);
	}

	struct svm_model* model;
	model = svm_train(&p.problem, &param);
	cout << "...[done]" << endl;

	if (svm_save_model(model_file_name.c_str(), model))
	{
		clog << "Can't save model to file " << model_file_name << endl;
		exit(1);
	}
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);

	return 0;
}
