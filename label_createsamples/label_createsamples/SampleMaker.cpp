#include "SampleMaker.h"
#include "Plane.h"
#include "Cylinder.h"
#include "ImageReader.h"
#include "InfoWriter.h"
#include "shading.h"
#include "opencv2/opencv.hpp"
#include "RandGen.h"

bool SampleMaker::ViewAngleTransform(cv::Mat src, cv::Mat dst, double xangle, double yangle, double zangle, int dist)
{
	cv::Size size = src.size();
	double w = (double)size.width, h = (double)size.height;

	// Projection 2D -> 3D matrix
	cv::Mat A1 = (cv::Mat_<double>(4, 3) <<
		1, 0, -w / 2,
		0, 1, -h / 2,
		0, 0, 0,
		0, 0, 1);

	// Rotation matrices around the X,Y,Z axis
	cv::Mat RX = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(xangle), -sin(xangle), 0,
		0, sin(xangle), cos(xangle), 0,
		0, 0, 0, 1);

	cv::Mat RY = (cv::Mat_<double>(4, 4) <<
		cos(yangle), 0, -sin(yangle), 0,
		0, 1, 0, 0,
		sin(yangle), 0, cos(yangle), 0,
		0, 0, 0, 1);

	cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
		cos(zangle), -sin(zangle), 0, 0,
		sin(zangle), cos(zangle), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// Composed rotation matrix with (RX,RY,RZ)
	cv::Mat R = RX * RY * RZ;

	// Translation matrix on the Z axis changes the distance
	cv::Mat T = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, (double)dist,
		0, 0, 0, 1);

	double f = (double)dist * 3;
	double u0 = w / 2, v0 = h / 2;
	double gamma = 0;
	// Camera intrinsic matrix 3D -> 2D
	// f = focal length
	// u0, v0 - principal point (optical axis)
	// gamma - skew coefficient between x and y
	cv::Mat K = (cv::Mat_<double>(3, 4) <<
		f, gamma, u0, 0,
		0, f, v0, 0,
		0, 0, 1, 0);

	// Final and overall transformation matrix
	cv::Mat transfo = K * (T * (R * A1));

	cv::warpPerspective(src, dst, transfo, size, cv::INTER_CUBIC | cv::WARP_INVERSE_MAP);

	return true;
}

bool SampleMaker::TransformImage(const cv::Mat& image)
{
	img_src = image;

	r = r * img_src.cols / PI;

	Object* object = 0;

	if (params.transformation == "" || params.transformation == "plane")
		object = new Plane;
	else if (params.transformation == "cylinder")
		object = new Cylinder;
	else
		CV_Error(CV_StsBadArg, "Unsupported transformation\n");
	
	object->surface.albedo = cv::Vec3d(object_albedo, object_albedo, object_albedo);
	object->surface.Ka = cv::Vec3d(object_Ka, object_Ka, object_Ka);
	object->surface.Kd = cv::Vec3d(object_Kd, object_Kd, object_Kd);
	object->surface.Ks = cv::Vec3d(object_Ks, object_Ks, object_Ks);
	object->surface.shininess = object_shininess;

	object->ProjectImage(img_src, r, i, transparent_color_low, transparent_color_high);
	object->Rotate(xangle, yangle, zangle);

	Scene scene;
	scene.AddObject(object);

	cv::Vec3d lmin, lmax;
	scene.GetLightingRange(*object, lmin, lmax);
	light_x = lmin[0] - params.light_dir_dev_max + (lmax[0] + params.light_dir_dev_max) * random->InRangeF(0, 1);
	light_y = lmin[1] - params.light_dir_dev_max + (lmax[1] + params.light_dir_dev_max) * random->InRangeF(0, 1);
	light_z = lmin[2] - params.light_dir_dev_max + (lmax[2] + params.light_dir_dev_max) * random->InRangeF(0, 1);

	auto light = new DistantLight(normalize(cv::Vec3d(light_x, light_y, light_z)), light_temp,
		cv::Vec3d(light_intensity, light_intensity, light_intensity));
	scene.AddLight(light);

	Plane frame;
	scene.Render(frame, true);

	if (params.grayscale)
		frame.OutMat(img_tran, alpha, CV_8UC1, bg_color, fill_color, false, params.noise_removal);
	else
		frame.OutMat(img_tran, alpha, CV_8UC3, bg_color, fill_color, false, params.noise_removal);

	return true;
}

void SampleMaker::PlaceTransformedImage(cv::Mat& bg)
{
	cv::Mat img, a;

	if (img_tran.channels() == 1)
		img.create(obj_height, obj_width, CV_8UC1);
	else
		img.create(obj_height, obj_width, CV_8UC3);

	a.create(obj_height, obj_width, CV_8UC1);
	resize(img_tran, img, img.size());
	resize(alpha, a, a.size());

	if (img_tran.channels() == 1)
	{
		for (int row = 0; row < img.rows; row++)
			for (int col = 0; col < img.cols; col++)
				if (a.at<uchar>(row, col))
					bg.at<uchar>(y + row + obj_y, x + col + obj_x) = img.at<uchar>(row, col);
	}
	else
	{
		for (int row = 0; row < img.rows; row++)
			for (int col = 0; col < img.cols; col++)
				if (a.at<uchar>(row, col))
					bg.at<cv::Vec3b>(y + row + obj_y, x + col + obj_x) = img.at<cv::Vec3b>(row, col);
	}
}

void SampleMaker::ProcessSample()
{
	transparent_color_low = params.transparent_color_low / 255.0;
	transparent_color_high = params.transparent_color_high / 255.0;

	for (int s = 0; s < params.num_patches; s++)
	{
		r = random->InRangeF(params.minrad, params.maxrad);
		xangle = random->InRangeF(-params.maxxangle, params.maxxangle);
		yangle = random->InRangeF(-params.maxyangle, params.maxyangle);
		zangle = random->InRangeF(-params.maxzangle, params.maxzangle);
		i = random->InRangeF(-params.maxincl, params.maxincl);

		object_albedo = random->InRangeF(params.albedo_min, params.albedo_max);
		object_Ka = params.ambient;
		object_Ks = random->InRangeF(params.specular_min, params.specular_max);
		object_Kd = 0.9 - object_Ks;
		object_shininess = random->InRangeF(params.shininess_min, params.shininess_max);
		light_temp = random->InRangeI(params.light_temp_min, params.light_temp_max);
		light_intensity = random->InRangeF(params.light_intensity_min, params.light_intensity_max);

		if (params.bg_color == "random")
			bg_color = random->InRangeF(0, 1);
		else if (params.bg_color == "transparent")
			bg_color = -1;
		else
			bg_color = atof(params.bg_color.c_str()) / 255.0;

		if (params.fill_color == "random")
			fill_color = random->InRangeF(0, 1);
		else if (params.fill_color == "transparent")
			fill_color = -1;
		else
			fill_color = atof(params.fill_color.c_str()) / 255.0;

		if (params.output_type == "patch")	// Just the background patch of fixed size (for negatives generation)
		{
			if (!params.patch_size)
				CV_Error(CV_StsBadArg, "Patch mode requires fixed size\n");

			patch_width = patch_height = params.patch_size;
			x = (int)(random->InRangeF(0.1, 0.9) * (bgreader.image.cols - patch_width));
			y = (int)(random->InRangeF(0.1, 0.9) * (bgreader.image.rows - patch_height));

			out_file_name = bgreader.imageshortname;
			infowriter->WriteImageEx(out_file_name, bgreader.image(cv::Rect(x, y, patch_width, patch_height)), patch_width, patch_height, ".jpg");
		}
		else
		{
			TransformImage(imreader.image);

			if (!params.patch_size)
			{
				patch_width = img_tran.cols - img_tran.cols % params.winsize;
				patch_height = img_tran.rows - img_tran.rows % params.winsize;
			}
			else
				patch_width = patch_height = params.patch_size;

			obj_width = (int)(std::min(patch_width, patch_height) * ((double)img_tran.size().width / (double)std::max(img_tran.size().width, img_tran.size().height)));
			obj_height = (int)(std::min(patch_width, patch_height) * ((double)img_tran.size().height / (double)std::max(img_tran.size().width, img_tran.size().height)));
			if (params.min_image_scale > 0)
			{
				obj_scale = random->InRangeF(params.min_image_scale, 1.0);
				obj_width = (int)(obj_width * obj_scale);
				obj_height = (int)(obj_height * obj_scale);
			}

			if (params.vary_image_pos)
			{
				obj_x = random->InRangeI(0, patch_width - obj_width);
				obj_y = random->InRangeI(0, patch_height - obj_height);
			}
			else
			{
				obj_x = (patch_width - obj_width) / 2;
				obj_y = (patch_height - obj_height) / 2;
			}

			if (params.output_type == "embed")	// Image is placed over background with predefined fill color and opacity 
			{
				x = (int)(random->InRangeF(0.1, 0.9) * (bgreader.image.cols - patch_width));
				y = (int)(random->InRangeF(0.1, 0.9) * (bgreader.image.rows - patch_height));


				char notes[200];
				sprintf_s(notes, sizeof(notes), "x=%d, y=%d, w=%d, h=%d, r=%.2f, xa=%.2f, ya=%.2f, za=%.2f, inlc=%.2f",
					x, y, patch_width, patch_height, r, xangle, yangle, zangle, i);
				cv::putText(bgreader.image, notes, cv::Point(10, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 255, 255), 1);
				sprintf_s(notes, sizeof(notes), "bgcolor=%.2f, fillcolor=%.2f, albedo=%.2f, Ka=%.2f, Kd=%.2f, Ks=%.2f, shin=%.2f",
					bg_color, fill_color, object_albedo, object_Ka, object_Kd, object_Ks, object_shininess);
				cv::putText(bgreader.image, notes, cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 255, 255), 1);
				sprintf_s(notes, sizeof(notes), "light_x=%.2f, light_y=%.2f, light_z=%.2f, temp=%d, intensity=%.2f",
					light_x, light_y, light_z, light_temp, light_intensity);
				cv::putText(bgreader.image, notes, cv::Point(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 255, 255), 1);

				PlaceTransformedImage(bgreader.image);

				out_file_name = bgreader.imageshortname;
				infowriter->WriteInfo(out_file_name, x, y, patch_width, patch_height, ".jpg");
			}
			else if (params.output_type == "crop" || params.output_type == "crop+patch")	// Image is placed over a background patch sized equal to image
			{
				if (bgreader.image.cols < patch_width || bgreader.image.rows < patch_height)
					CV_Error(CV_StsBadArg, "background image is smaller than the patch size\n");

				x = (int)(random->InRangeF(0.1, 0.9) * (bgreader.image.cols - patch_width));
				y = (int)(random->InRangeF(0.1, 0.9) * (bgreader.image.rows - patch_height));

				out_file_name = bgreader.imageshortname;
				std::string filename = infowriter->MakeFileName(out_file_name, patch_width, patch_height, ".jpg");

				if (params.output_type == "crop+patch")	// Additionally write an empty patch (negative image)
					infowriter->WriteImage(filename, params.output_dir_bg, bgreader.image(cv::Rect(x, y, patch_width, patch_height)));

				PlaceTransformedImage(bgreader.image);

				infowriter->WriteImage(filename, params.output_dir, bgreader.image(cv::Rect(x, y, patch_width, patch_height)));
				infowriter->WriteInfo(filename, x, y, patch_width, patch_height, ".jpg");

			}
			else if (params.output_type == "nobg")	// No background, image is placed as is
			{
				x = 0;
				y = 0;

				cv::Mat bg;
				if (img_tran.channels() == 1)
					bg.create(patch_height, patch_width, CV_8UC1);
				else
					bg.create(patch_height, patch_width, CV_8UC3);

				PlaceTransformedImage(bg);

				out_file_name = imreader.imageshortname;
				infowriter->WriteImageEx(out_file_name, bg, patch_width, patch_height, ".jpg");
			}
		}
	}

	if (params.output_type == "embed")	// Write it only when all patches are placed 
	{
		infowriter->WriteImageEx(out_file_name, bgreader.image, patch_width, patch_height, ".jpg");
	}
}


void SampleMaker::Visualize(std::string imagename)
{
	img_src = cv::imread(imagename, cv::IMREAD_COLOR);
	if (img_src.empty())
		CV_Error(CV_StsBadArg, "Error opening the source image");

	cv::Mat img(std::max(img_src.rows, img_src.cols) + 100, std::max(img_src.rows, img_src.cols) + 100, CV_8UC3);

	double rad = 3.0;
	i = 0;
	transparent_color_low = params.transparent_color_low / 255.0;
	transparent_color_high = params.transparent_color_high / 255.0;
	bg_color = 255;
	fill_color = -1;
	xangle = yangle = zangle = 0;
	object_albedo = 0.5;
	object_shininess = 40;
	object_Ka = 0.1;
	object_Kd = 0.5;
	object_Ks = 0.4;
	light_temp = 5000;
	light_intensity = 1;
	light_x = 0;
	light_y = 0;
	light_z = 1;

	int bg_color_i = 255, fill_color_i = -1;

	while (1)
	{
		r = rad * img_src.cols / PI;
		bg_color = bg_color_i / 255.0;
		fill_color = fill_color_i / 255.0;

		auto label = new Cylinder;
		label->ProjectImage(img_src, r, i, transparent_color_low, transparent_color_high);
		label->Rotate(xangle, yangle, zangle);
		label->surface.albedo = cv::Vec3d(object_albedo, object_albedo, object_albedo);
		label->surface.Ka = cv::Vec3d(object_Ka, object_Ka, object_Ka);
		label->surface.Kd = cv::Vec3d(object_Kd, object_Kd, object_Kd);
		label->surface.Ks = cv::Vec3d(object_Ks, object_Ks, object_Ks);
		label->surface.shininess = object_shininess;

		auto light = new DistantLight(normalize(cv::Vec3d(light_x, light_y, light_z)), light_temp,
			cv::Vec3d(light_intensity, light_intensity, light_intensity));

		Scene scene;
		scene.AddObject(label);
		scene.AddLight(light);

		Plane frame;
		scene.Render(frame);

		frame.OutMat(img_tran, alpha, CV_8UC3, cv::Vec3d(bg_color, bg_color, bg_color), cv::Vec3d(fill_color, fill_color, fill_color),
			false, params.noise_removal);
		img.setTo(cv::Scalar(0));
		img_tran.copyTo(img(cv::Rect(50, 50, img_tran.cols, img_tran.rows)));

		if (!cascade.empty())
		{
			cv::Mat image_grey;
			cvtColor(img, image_grey, cv::COLOR_BGR2GRAY);

			std::vector<cv::Rect> detected_areas;
			std::vector<int> reject_levels;
			std::vector<double> level_weights;
			cv::CascadeClassifier cascade(cascade);

			cascade.detectMultiScale(image_grey, detected_areas);

			for (int i = 0; i < detected_areas.size(); i++)
			{
				cv::Rect LogoRect = detected_areas[i];
				rectangle(img, LogoRect, cv::Scalar(255, 0, 255), 2, cv::LINE_4);
			}
		}

		printf("(r) radius = %f\n"
			"(i) inclination = %f\n"
			"(x) x angle = %f\n"
			"(y) y angle = %f\n"
			"(z) z angle = %f\n"
			"(b) backgound color = %d\n"
			"(f) backgound fill color = %d\n"
			"(l) albedo = %f\n"
			"(a) K ambient = %f\n"
			"(d) K diffuse = %f\n"
			"(s) K specular = %f\n"
			"(S) shininess = %d\n"
			"(C) light temperature = %d\n"
			"(I) light intensity = %f\n"
			"(X) light x = %f\n"
			"(Y) light y = %f\n"
			"(Z) light z = %f\n"
			"(D) detect with cascade = %s\n",
			rad, i, xangle, yangle, zangle, (int)(bg_color * 255), (int)(fill_color * 255),
			object_albedo, object_Ka, object_Kd, object_Ks,
			(int)object_shininess, light_temp, light_intensity, light_x, light_y, light_z,
			cascade.c_str());
		printf("Press a key to change transformation parameters.\n");
		imshow("shape", img);
		int key = cv::waitKey(0);
		printf("%d\n", key);

		if (key == 113)	// q
		{
			break;
		}
		else if (key == 114) // r
		{
			printf("Enter radius (1-inf): ");
			scanf_s("%lf", &rad);
		}
		else if (key == 105) // i
		{
			printf("Enter inclination (-1.5-+1.5): ");
			scanf_s("%lf", &i);
		}
		else if (key == 120) // x
		{
			printf("Enter x angle (-1.5-+1.5): ");
			scanf_s("%lf", &xangle);
		}
		else if (key == 121) // y
		{
			printf("Enter y angle (-1.5-+1.5): ");
			scanf_s("%lf", &yangle);
		}
		else if (key == 122) // z
		{
			printf("Enter z angle (-1.5-+1.5): ");
			scanf_s("%lf", &zangle);
		}
		else if (key == 98) // b
		{
			printf("Enter background color (-1-+255): ");
			scanf_s("%d", &bg_color_i);
		}
		else if (key == 102) // f
		{
			printf("Enter background fill color (-1-+255): ");
			scanf_s("%d", &fill_color_i);
		}
		else if (key == 108) // l
		{
			printf("Enter albedo (0-1): ");
			scanf_s("%lf", &object_albedo);
		}
		else if (key == 97) // a
		{
			printf("Enter K ambient (0-1): ");
			scanf_s("%lf", &object_Ka);
			object_Kd = 1 - object_Ka - object_Ks;
		}
		else if (key == 100) // d
		{
			printf("Enter K diffuse (0-1): ");
			scanf_s("%lf", &object_Kd);
			object_Ks = 1 - object_Ka - object_Kd;
		}
		else if (key == 115) // s
		{
			printf("Enter K specular (0-1): ");
			scanf_s("%lf", &object_Ks);
			object_Kd = 1 - object_Ka - object_Ks;
		}
		else if (key == 83) // S
		{
			printf("Enter shininess (0-inf): ");
			scanf_s("%lf", &object_shininess);
		}
		else if (key == 67) // C
		{
			printf("Enter light temperature (1000-40000): ");
			scanf_s("%d", &light_temp);
		}
		else if (key == 73) // I
		{
			printf("Enter light intensity (1-inf): ");
			scanf_s("%lf", &light_intensity);
		}
		else if (key == 88) // X
		{
			printf("Enter light x direction (inf-+inf): ");
			scanf_s("%lf", &light_x);
		}
		else if (key == 89) // Y
		{
			printf("Enter light y direction (-inf-+inf): ");
			scanf_s("%lf", &light_y);
		}
		else if (key == 90) // Z
		{
			printf("Enter light z direction (-inf-+inf): ");
			scanf_s("%lf", &light_z);
		}
		else if (key == 68) // D
		{
			char tmp[_MAX_PATH];
			printf("Enter cascade path: ");
			scanf_s("%s", &tmp, (unsigned)_countof(tmp));
			cascade = tmp;
		}
	}
}