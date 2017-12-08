#define _CRT_SECURE_NO_WARNINGS

#include "utility.h"
#include "Plane.h"
#include "Cylinder.h"
#include "BackgroundImageReader.h"
#include "InfoWriter.h"
#include "shading.h"

bool ViewAngleTransform(Mat src, Mat dst, double xangle, double yangle, double zangle, int dist)
{
	Size size = src.size();
	double w = (double)size.width, h = (double)size.height;

	// Projection 2D -> 3D matrix
	Mat A1 = (Mat_<double>(4, 3) <<
		1, 0, -w/2,
		0, 1, -h/2,
		0, 0, 0,
		0, 0, 1);

	// Rotation matrices around the X,Y,Z axis
	Mat RX = (Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(xangle), -sin(xangle), 0,
		0, sin(xangle), cos(xangle), 0,
		0, 0, 0, 1);

	Mat RY = (Mat_<double>(4, 4) <<
		cos(yangle), 0, -sin(yangle), 0,
		0, 1, 0, 0,
		sin(yangle), 0, cos(yangle), 0,
		0, 0, 0, 1);

	Mat RZ = (Mat_<double>(4, 4) <<
		cos(zangle), -sin(zangle), 0, 0,
		sin(zangle), cos(zangle), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// Composed rotation matrix with (RX,RY,RZ)
	Mat R = RX * RY * RZ;

	// Translation matrix on the Z axis changes the distance
	Mat T = (Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, (double)dist,
		0, 0, 0, 1);

	double f = (double)dist*3;
	double u0 = w / 2, v0 = h / 2;
	double gamma = 0;
	// Camera intrinsic matrix 3D -> 2D
	// f = focal length
	// u0, v0 - principal point (optical axis)
	// gamma - skew coefficient between x and y
	Mat K = (Mat_<double>(3, 4) <<
		f, gamma, u0, 0,
		0, f, v0, 0,
		0, 0, 1, 0);

	// Final and overall transformation matrix
	Mat transfo = K * (T * (R * A1));

	warpPerspective(src, dst, transfo, size, INTER_CUBIC | WARP_INVERSE_MAP);

	return true;
}

bool TransformImage(string imagename, ImageTransformData* data)
{
	data->img_src = imread(imagename, IMREAD_GRAYSCALE);
	if (data->img_src.empty())
		CV_Error(CV_StsBadArg, "Error opening the source image");

	data->r = data->r * data->img_src.cols / PI;

	auto label = new Cylinder;
	label->surface.albedo = Vec3d(data->object_albedo, data->object_albedo, data->object_albedo);
	label->surface.Ka = Vec3d(data->object_Ka, data->object_Ka, data->object_Ka);
	label->surface.Kd = Vec3d(data->object_Kd, data->object_Kd, data->object_Kd);
	label->surface.Ks = Vec3d(data->object_Ks, data->object_Ks, data->object_Ks);
	label->surface.shininess = data->object_shininess;

	label->ProjectImage8(data->img_src, data->r, data->i, data->transparent_color_low, data->transparent_color_high);
	label->Rotate(data->xangle, data->yangle, data->zangle);

	//label->OutMat(data->img_tran, CV_8U, Vec3d(data->bg_fill_color, data->bg_fill_color, data->bg_fill_color), false, true, true);
	//Object::RemoveAlpha(data->img_tran, data->img_tran);
	//imshow("frame", data->img_tran);
	//waitKey(0);

	Scene scene;
	scene.AddObject(label);

	cv::Vec3d lmin, lmax;
	scene.GetLightingRange(*label, lmin, lmax);
	data->light_x = lmin[0] - data->params.light_dir_dev_max + (lmax[0] + data->params.light_dir_dev_max) * ((double)rand() / RAND_MAX);
	data->light_y = lmin[1] - data->params.light_dir_dev_max + (lmax[1] + data->params.light_dir_dev_max) * ((double)rand() / RAND_MAX);
	data->light_z = lmin[2] - data->params.light_dir_dev_max + (lmax[2] + data->params.light_dir_dev_max) * ((double)rand() / RAND_MAX);

	auto light = new DistantLight(normalize(Vec3d(data->light_x, data->light_y, data->light_z)), Vec3d(data->light_color, data->light_color, data->light_color),
		Vec3d(data->light_intensity, data->light_intensity, data->light_intensity));
	scene.AddLight(light);

	Plane frame;
	scene.Render(frame, true);

	frame.OutMat(data->img_tran, CV_8U, data->bg_fill_color, false, true, data->params.noise_removal);
//	label->OutMat(data->img_tran, data->bg_fill_color, true, true, true);

	//imshow("frame", data->img_tran);
	//waitKey(0);

	//GaussianBlur(data->img_tran, data->img_tran, Size(3, 3), 0, 0);
	////GaussianBlur(data->mask, data->mask, Size(3, 3), 0, 0);

	//imshow("img_tran", data->img_tran);
	//waitKey(0);

	return true;
}

bool PlaceTransformedImage(Mat background, ImageTransformData* data)
{
	Mat img, mask;
	uchar *pbg, main, alpha;

	img.create(data->height, data->width, CV_8UC1);

	resize(data->img_tran, img, img.size());

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			main = img.at<Vec2b>(row, col)[0];
			pbg = &background.at<unsigned char>(data->y + row, data->x + col);
			alpha = img.at<Vec2b>(row, col)[1];
			if (alpha)
				*pbg = main;
		}
	}
	//imshow("background", background);
	//waitKey(0);

	return true;
}

void CreateTestSamples(string infoname,
	string imagename,
	string bgname,
	ImageTransformData* data)
{
	BackgroundImageReader bgreader;
	InfoWriter infowriter;

	int i, pos;
	
	if (infoname.empty() || imagename.empty() || bgname.empty())
		CV_Error(CV_StsBadArg, "infoname or imagename or bgname is NULL\n");

	if (!bgreader.Create(bgname))
		CV_Error(CV_StsBadArg, "Error opening background file list\n");

	if (!infowriter.Create(infoname))
		CV_Error(CV_StsBadArg, "Error opening info file\n");
	
	data->transparent_color_low = data->params.transparent_color_low / 255.0;
	data->transparent_color_high = data->params.transparent_color_high / 255.0;

	if (data->params.numsample > 0 && data->params.numsample < bgreader.count)
		bgreader.count = data->params.numsample;

	for (i = 0; i < bgreader.count; i++)
	{
		if (data->params.random)
			pos = bgreader.NextRandom();
		else
			pos = bgreader.Next();

		data->bgname = bgreader.imageshortname;

		if ((data->maxscale = data->params.maxscale) < 0.0)
		{
			data->maxscale = MIN(0.5 * bgreader.image.cols / data->params.winsize,
				0.5 * bgreader.image.rows / data->params.winsize);
		}
		if (data->maxscale < 1.0)
			CV_Error(CV_StsBadArg, "Image is too small for window.\n");

		data->r = data->params.minrad + ((double)rand() / RAND_MAX) * (data->params.maxrad - data->params.minrad);
		data->xangle = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxxangle - data->params.maxxangle;
		data->yangle = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxyangle - data->params.maxyangle;
		data->zangle = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxzangle - data->params.maxzangle;
		data->i = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxincl - data->params.maxincl;

		data->object_albedo = data->params.albedo_min + ((double)rand() / RAND_MAX) * (data->params.albedo_max - data->params.albedo_min);
		data->object_Ka = data->params.ambient;
		data->object_Ks = data->params.specular_min + ((double)rand() / RAND_MAX) * (data->params.specular_max - data->params.specular_min);
		data->object_Kd = 0.9 - data->object_Ks;
		data->object_shininess = data->params.shininess_min + ((double)rand() / RAND_MAX) * (data->params.shininess_max - data->params.shininess_min);
		data->light_color = data->params.light_color;
		data->light_intensity = data->params.light_intensity_min + ((double)rand() / RAND_MAX) * (data->params.light_intensity_max - data->params.light_intensity_min);

		if (data->params.bg_fill_color == "random")
			data->bg_fill_color = ((double)rand() / RAND_MAX);
		else if (data->params.bg_fill_color == "transparent")
			data->bg_fill_color = -1;
		else
			data->bg_fill_color = atof(data->params.bg_fill_color.c_str()) / 255;

		//data->r = 4;
		//data->i = 0.2;
		//data->yangle = 0.1;
		//data->xangle = 0.8;
		//data->zangle = 0;

		//data->light_x = -1;
		//data->light_y = 0;
		//data->light_color = 1;
		//data->light_intensity = 2;

		printf("Processing background image #%d: %s\n", pos, bgreader.imagefullname.c_str());
		
		TransformImage(imagename, data);

		data->scale = (data->maxscale - 1.0) * rand() / RAND_MAX + 1.0;
		if (data->img_tran.cols >= data->img_tran.rows)
		{
			data->height = (int)(data->scale * data->params.winsize);
			data->width = (int)((data->scale * data->params.winsize) * ((float)data->img_tran.cols / (float)data->img_tran.rows));
		}
		else
		{
			data->width = (int)(data->scale * data->params.winsize);
			data->height = (int)((data->scale * data->params.winsize) * ((float)data->img_tran.rows / (float)data->img_tran.cols));
		}
		data->x = (int)((0.1 + 0.8 * rand() / RAND_MAX) * (bgreader.image.cols - data->width));
		data->y = (int)((0.1 + 0.8 * rand() / RAND_MAX) * (bgreader.image.rows - data->height));

		char notes[200];
		sprintf(notes, "x=%d, y=%d, w=%d, h=%d, r=%.2f, xa=%.2f, ya=%.2f, za=%.2f, inlc=%.2f",
			data->x, data->y, data->width, data->height, data->r, data->xangle, data->yangle, data->zangle, data->i);
		putText(bgreader.image, notes, Point(10, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 255, 255), 1);
		sprintf(notes, "bgcolor=%.2f, albedo=%.2f, Ka=%.2f, Kd=%.2f, Ks=%.2f, shin=%.2f",
			data->bg_fill_color, data->object_albedo, data->object_Ka, data->object_Kd, data->object_Ks, data->object_shininess);
		putText(bgreader.image, notes, Point(10, 25), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 255, 255), 1);
		sprintf(notes, "light_x=%.2f, light_y=%.2f, light_z=%.2f, color=%.2f, intensity=%.2f",
			data->light_x, data->light_y, data->light_z, data->light_color, data->light_intensity);
		putText(bgreader.image, notes, Point(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 255, 255), 1);

		PlaceTransformedImage(bgreader.image, data);

		infowriter.WriteInfo(i, data, ".jpg");
		infowriter.WriteImage(i, data, bgreader.image, ".jpg");
	}
}

void Visualize(string imagename, ImageTransformData* data)
{
	data->img_src = imread(imagename, IMREAD_COLOR);
	if (data->img_src.empty())
		CV_Error(CV_StsBadArg, "Error opening the source image");

	Mat img(std::max(data->img_src.rows, data->img_src.cols)+100, std::max(data->img_src.rows, data->img_src.cols)+100, CV_8UC3);

	double r = 3;
	data->i = 0;
	data->transparent_color_low = data->params.transparent_color_low / 255.0;
	data->transparent_color_high = data->params.transparent_color_high / 255.0;
	int bg_fill_color = -1;
	data->xangle = data->yangle = data->zangle = 0;
	data->object_albedo = 0.5;
	data->object_shininess = 40;
	data->object_Ka = 0.1;
	data->object_Kd = 0.5;
	data->object_Ks = 0.4;
	data->light_color = 255;
	data->light_intensity = 1;
	data->light_x = 0;
	data->light_y = 0;
	data->light_z = 1;

	while (1)
	{
		data->r = r * data->img_src.cols / PI;
		data->bg_fill_color = bg_fill_color / 255.0;

		auto label = new Cylinder;
		label->ProjectImageRGB(data->img_src, data->r, data->i, data->transparent_color_low, data->transparent_color_high);
		label->Rotate(data->xangle, data->yangle, data->zangle);

		label->surface.albedo = Vec3d(data->object_albedo, data->object_albedo, data->object_albedo);
		label->surface.Ka = Vec3d(data->object_Ka, data->object_Ka, data->object_Ka);
		label->surface.Kd = Vec3d(data->object_Kd, data->object_Kd, data->object_Kd);
		label->surface.Ks = Vec3d(data->object_Ks, data->object_Ks, data->object_Ks);
		label->surface.shininess = data->object_shininess;
		Scene scene;
		scene.AddObject(label);

		auto light = new DistantLight(normalize(Vec3d(data->light_x, data->light_y, data->light_z)), Vec3d(data->light_color/255, data->light_color/255, data->light_color/255),
			Vec3d(data->light_intensity, data->light_intensity, data->light_intensity));
		scene.AddLight(light);

		Plane frame;
		scene.Render(frame);

		frame.OutMat(data->img_tran, CV_32S, data->bg_fill_color, false, true, data->params.noise_removal);
		img.setTo(cv::Scalar(0));
		Object::RemoveAlpha(data->img_tran, data->img_tran);
		data->img_tran.copyTo(img(Rect(50, 50, data->img_tran.cols, data->img_tran.rows)));

		printf("(r) radius = %f\n"
			"(i) inclination = %f\n"
			"(x) x angle = %f\n"
			"(y) y angle = %f\n"
			"(z) z angle = %f\n"
			"(l) albedo = %f\n"
			"(a) K ambient = %f\n"
			"(d) K diffuse = %f\n"
			"(s) K specular = %f\n"
			"(S) shininess = %d\n"
			"(C) light color = %d\n"
			"(I) light intensity = %f\n"
			"(X) light x = %f\n"
			"(Y) light y = %f\n"
			"(Z) light z = %f\n",
			r, data->i, data->xangle, data->yangle, data->zangle, data->object_albedo, data->object_Ka, data->object_Kd, data->object_Ks,
			(int)data->object_shininess, (int)data->light_color, data->light_intensity, data->light_x, data->light_y, data->light_z);
		printf("Press a key to change transformation parameters.\n");
		imshow("shape", img);
		int key = waitKey(0);
		printf("%d\n", key);
		
		if (key == 113)	// q
		{
			break;
		}
		else if (key == 114) // r
		{
			printf("Enter radius (1-inf): ");
			scanf("%lf", &r);
		}
		else if (key == 105) // i
		{
			printf("Enter inclination (-1.5-+1.5): ");
			scanf("%lf", &data->i);
		}
		else if (key == 120) // x
		{
			printf("Enter x angle (-1.5-+1.5): ");
			scanf("%lf", &data->xangle);
		}
		else if (key == 121) // y
		{
			printf("Enter y angle (-1.5-+1.5): ");
			scanf("%lf", &data->yangle);
		}
		else if (key == 122) // z
		{
			printf("Enter z angle (-1.5-+1.5): ");
			scanf("%lf", &data->zangle);
		}
		else if (key == 108) // l
		{
			printf("Enter albedo (0-1): ");
			scanf("%lf", &data->object_albedo);
		}
		else if (key == 97) // a
		{
			printf("Enter K ambient (0-1): ");
			scanf("%lf", &data->object_Ka);
			data->object_Kd = 1 - data->object_Ka - data->object_Ks;
		}
		else if (key == 100) // d
		{
			printf("Enter K diffuse (0-1): ");
			scanf("%lf", &data->object_Kd);
			data->object_Ks = 1 - data->object_Ka - data->object_Kd;
		}
		else if (key == 115) // s
		{
			printf("Enter K specular (0-1): ");
			scanf("%lf", &data->object_Ks);
			data->object_Kd = 1 - data->object_Ka - data->object_Ks;
		}
		else if (key == 83) // S
		{
			printf("Enter shininess (0-inf): ");
			scanf("%lf", &data->object_shininess);
		}
		else if (key == 67) // C
		{
			printf("Enter light color (1-255): ");
			scanf("%lf", &data->light_color);
		}
		else if (key == 73) // I
		{
			printf("Enter light intensity (1-inf): ");
			scanf("%lf", &data->light_intensity);
		}
		else if (key == 88) // X
		{
			printf("Enter light x direction (inf-+inf): ");
			scanf("%lf", &data->light_x);
		}
		else if (key == 89) // Y
		{
			printf("Enter light y direction (-inf-+inf): ");
			scanf("%lf", &data->light_y);
		}
		else if (key == 90) // Z
		{
			printf("Enter light z direction (-inf-+inf): ");
			scanf("%lf", &data->light_z);
		}
	}
}