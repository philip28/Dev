#define _CRT_SECURE_NO_WARNINGS

#include "utility.h"
#include "Plane.h"
#include "Cylinder.h"
#include "BackgroundImageReader.h"
#include "InfoWriter.h"
#include "shading.h"


inline int _getcylx(double r, double phi, int l)
{
	double alpha, dx;

	alpha = atan2(l, r);
	dx = r * sin(phi + alpha);
	
	return (int)round(dx);
}

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

bool WrapConeTransform3D_(Mat src, Mat &dst, double r1, double r2, double phi, unsigned int bgcolor)
{
	double cx, r, dr;
	double alpha, dx;
	int minx, maxx;
	int y, l;
	int dst_width, dst_height;

	if (src.depth() != CV_8U || src.channels() != 1 || src.dims != 2)
	{
		CV_Error(CV_StsBadArg, "Source must be a two-dimensional array of CV_8UC1 type.");
		return false;
	}

	cx = (double)src.cols / 2;
	minx = MIN(MIN(_getcylx(r1, phi, -(int)cx), _getcylx(r2, phi, -(int)cx)), _getcylx((r1 + r2) / 2, phi, -(int)cx));
	maxx = MAX(MAX(_getcylx(r1, phi, (int)cx - 1), _getcylx(r2, phi, (int)cx - 1)), _getcylx((r1 + r2) / 2, phi, (int)cx - 1));

	// initialize dest image
	dst_width = abs(maxx - minx) + 1;
	dst_height = src.rows;
	dst.create(dst_height, dst_width, CV_8UC1);
	dst.setTo(Scalar(bgcolor));

	int size[] = { src.cols, src.rows, ((int)MAX(r1, r2)+1) * 2 };
	SparseMat obj3d(3, size, CV_8UC1);

	dr = (r2 - r1) / dst_height;
	r = r1;

	for (y = 0; y < dst_height; y++)
	{
		for (l = 0; l < (int)cx; l++)
		{
			alpha = atan2(l, r);
			dx = r * sin(phi + alpha);
			*obj3d.ptr((int)round(dx) - minx, y, (int)round(r * (1 - cos(alpha))), true) = src.at<uchar>(y, (int)(cx + l)); // "left" of new central point

			alpha = atan2(-l, r);
			dx = r * sin(phi + alpha);
			*obj3d.ptr((int)round(dx) - minx, y, (int)round(r * (1 - cos(alpha))), true) = src.at<uchar>(y, (int)(cx - l)); // "right" of new central point
		}

		// In case of an even width
		if (!(src.cols % 2))
		{
			alpha = atan2(-l, r);
			dx = r * sin(phi + alpha);
			*obj3d.ptr((int)round(dx) - minx, y, (int)round(r * (1 - cos(alpha))), true) = src.at<uchar>(y, 0);
		}

		r += dr;
	}

	// Ugly Z-buffer filtering
	typedef struct {
		int x, y, z;
		uchar value;
	} point3d;

	SparseMatIterator_<uchar> its, its_end;
	vector<point3d> vec3d;
	point3d point;

	its = obj3d.begin<uchar>();
	its_end = obj3d.end<uchar>();
	for (; its != its_end; ++its)
	{
		point.x = its.node()->idx[0];
		point.y = its.node()->idx[1];
		point.z = its.node()->idx[2];
		point.value = its.value<uchar>();
		vec3d.push_back(point);
	}

	std::sort(vec3d.begin(), vec3d.end(), [](point3d a, point3d b) {
		return a.z > b.z;
	});

	vector<point3d>::iterator itv, itv_end;
	itv = vec3d.begin();
	itv_end = vec3d.end();

	for (; itv != itv_end; ++itv)
		dst.at<uchar>((*itv).y, (*itv).x) = (*itv).value;

	imshow("3D", dst);
	waitKey(0);


	return true;
}

bool TransformImage(string imagename, ImageTransformData* data)
{
//	Mat img_erode, img_dilate;
	uchar *pmask;
//	uchar *psrc, *perode, *pdilate;
//	uchar dd, de;

	data->img_src = imread(imagename.c_str(), IMREAD_GRAYSCALE);
	if (data->img_src.empty())
		CV_Error(CV_StsBadArg, "Error opening the source image");

	data->r = data->r * data->img_src.cols / PI;

	auto label = new Cylinder;
	label->surface.albedo = Vec3d(data->object_albedo, data->object_albedo, data->object_albedo);
	label->surface.Ka = Vec3d(data->object_Ka, data->object_Ka, data->object_Ka);
	label->surface.Kd = Vec3d(data->object_Kd, data->object_Kd, data->object_Kd);
	label->surface.Ks = Vec3d(data->object_Ks, data->object_Ks, data->object_Ks);
	label->surface.shininess = data->object_shininess;

	label->ProjectImage8(data->img_src, data->r, data->i, data->params.bgcolor);
	label->Rotate(data->xangle, data->yangle, data->zangle);

	//label->OutMat(data->img_tran, data->params.bgcolor);
	//imshow("frame", data->img_tran);
	//waitKey(0);

	Scene scene;
	scene.AddObject(label);

	auto light = new DistantLight(normalize(Vec3d(data->light_x, data->light_y, 1)), Vec3d(data->light_color, data->light_color, data->light_color),
		Vec3d(data->light_intensity, data->light_intensity, data->light_intensity));
	scene.AddLight(light);

	Plane frame;
	scene.Render(frame);

	frame.OutMat(data->img_tran, data->params.bgcolor);

	//imshow("frame", data->img_tran);
	//waitKey(0);

	data->mask = data->img_tran.clone();
//	img_erode = data->img_tran.clone();
//	img_dilate = data->img_tran.clone();

	for (int row = 0; row < data->mask.rows; row++)
	{
		for (int col = 0; col < data->mask.cols; col++)
		{
			pmask = &data->mask.at<unsigned char>(row, col);
			if (*pmask <= data->params.bgthreshold && *pmask >= data->params.bgcolor)
				*pmask = 0;
			else
				*pmask = 255;
		}
	}

	//erode(data->img_tran, img_erode, Mat(), Point(1,1), 1, BORDER_REPLICATE);
	//dilate(data->img_tran, img_dilate, Mat(), Point(1, 1), 1, BORDER_REPLICATE);

	//for (int row = 0; row < data->mask.rows; row++)
	//{
	//	for (int col = 0; col < data->mask.cols; col++)
	//	{
	//		pmask = &data->mask.at<uchar>(row, col);
	//		if ((*pmask) == 0)
	//		{
	//			psrc = &data->img_tran.at<uchar>(row, col);
	//			perode = &img_erode.at<uchar>(row, col);
	//			pdilate = &img_dilate.at<uchar>(row, col);
	//			de = data->params.bgcolor - *perode;
	//			dd = *pdilate - data->params.bgcolor;
	//			if (de >= dd && de > data->params.bgthreshold)
	//			{
	//				*psrc = *perode;
	//			}
	//			if (dd > de && dd > data->params.bgthreshold)
	//			{
	//				*psrc = *pdilate;
	//			}
	//		}
	//	}
	//}


//	GaussianBlur(data->img_tran, data->img_tran, Size(3, 3), 0, 0);
	GaussianBlur(data->mask, data->mask, Size(3, 3), 0, 0);

	//imshow("img_tran", data->img_tran);
	//waitKey(0);

	return true;
}

bool PlaceTransformedImage(Mat background, ImageTransformData* data)
{
	Mat img, mask;
	unsigned char *pimg, *pbg, *palpha;
	unsigned char chartmp;
	int forecolordev;

	img.create(data->height, data->width, CV_8UC1);
	mask.create(data->height, data->width, CV_8UC1);

	resize(data->img_tran, img, img.size());
	resize(data->mask, mask, mask.size());

	forecolordev = (int)(data->params.maxintensitydev * (2.0 * rand() / RAND_MAX - 1.0));
	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			pimg = &img.at<unsigned char>(row, col);
			pbg = &background.at<unsigned char>(data->y + row, data->x + col);
			palpha = &mask.at<unsigned char>(row, col);
			chartmp = (uchar)MAX(0, MIN(255, forecolordev + (*pimg)));

			*pbg = (uchar)((chartmp * (*palpha) + (255 - (*palpha)) * (*pbg)) / 255);
		}
	}
	//imshow("background", background);
	//waitKey(0);

	return true;
}

//static double object_albedo = 0.1;
//static double object_Ka = 0.1;
//static double object_Kd = 0.6;
//static double object_Ks = 0.3;
//static double object_spec_n = 20;
//static int light_intensity;
//
//
//void CreateTestSamples(string infoname,
//	string imagename,
//	string bgname,
//	ImageTransformData* data)
//{
//	BackgroundImageReader bgreader;
//	InfoWriter infowriter;
//
//	int i = 0, pos = 0;
//
//	if (infoname.empty() || imagename.empty() || bgname.empty())
//		CV_Error(CV_StsBadArg, "infoname or imagename or bgname is NULL\n");
//
//	if (!bgreader.Create(bgname))
//		CV_Error(CV_StsBadArg, "Error opening background file list\n");
//
//	if (!infowriter.Create(infoname))
//		CV_Error(CV_StsBadArg, "Error opening info file\n");
//
//	bgreader.Next();
//	for (object_albedo = 0; object_albedo <= 1; object_albedo += 0.1) {
//		for (object_Kd = 0; object_Kd <= 0.9; object_Kd += 0.1) {
//			object_Ks = 0.9 - object_Kd;
//			for (object_spec_n = 10; object_spec_n <= 40; object_spec_n += 5) {
//				for (light_intensity = 1; light_intensity < 4; light_intensity++) {
//					pos++;
//					i++;
//
//					if (data->params.maxscale < 0.0)
//					{
//						data->params.maxscale = MIN(0.7F * bgreader.image.cols / data->params.winwidth,
//							0.7F * bgreader.image.rows / data->params.winheight);
//					}
//					if (data->params.maxscale < 1.0F)
//						CV_Error(CV_StsBadArg, "Scaling down not supported\n");
//
//					data->r = data->params.minrad + ((double)rand() / RAND_MAX) * (data->params.maxrad - data->params.minrad);
//					data->xangle = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxxangle - data->params.maxxangle;
//					data->yangle = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxyangle - data->params.maxyangle;
//					data->zangle = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxzangle - data->params.maxzangle;
//					data->i = (double)rand() / (RAND_MAX + 1) * 2 * data->params.maxincl - data->params.maxincl;
//
//					//data->r = 3;
//					//data->i = 0;
//					//data->yangle = 0;
//					//data->xangle = 0;
//					//data->zangle = 0;
//
//					data->light_x = 0;
//					data->light_y = 0;
//					data->light_color = 1;
//					data->light_intensity = light_intensity;
//
//					data->object_albedo = object_albedo;
//					data->object_Ka = object_Ka;
//					data->object_Kd = object_Kd;
//					data->object_Ks = object_Ks;
//					data->object_spec_n = object_spec_n;
//
//					printf("Processing background image #%d: %s\n", pos, bgreader.imagename.c_str());
//
//					TransformImage(imagename, data);
//
//					data->scale = 0;
//					data->height = data->img_tran.rows;
//					data->width = data->img_tran.cols;
//					data->x = 500;
//					data->y = 500;
//
//					Mat image = bgreader.image.clone();
//					PlaceTransformedImage(image, data);
//
//					char sss[200];
//					sprintf(sss, "albedo = %f, Ka = %f, Kd = %f, Ks = %f, n = %d, intensity = %d", data->object_albedo, data->object_Ka, data->object_Kd, data->object_Ks, (int)data->object_spec_n, (int)data->light_intensity);
//					putText(image, sss, Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 2);
//
//					infowriter.WriteImage(i, data, image);
//				}
//			}
//		}
//	}
//}


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
	
	for (i = 0; i < bgreader.count; i++)
	{
		if (data->params.random)
			pos = bgreader.NextRandom();
		else
			pos = bgreader.Next();

		if (data->params.maxscale < 0.0)
		{
			data->params.maxscale = MIN(0.5F * bgreader.image.cols / data->params.winwidth,
				0.5F * bgreader.image.rows / data->params.winheight);
		}
		if (data->params.maxscale < 1.0F)
			CV_Error(CV_StsBadArg, "Scaling down not supported\n");

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
		data->light_x = tan(-data->params.light_x_max + ((double)rand() / RAND_MAX) * data->params.light_x_max);
		data->light_y = tan(-data->params.light_y_max + ((double)rand() / RAND_MAX) * data->params.light_y_max);

		//data->r = 3;
		//data->i = 0;
		//data->yangle = 0;
		//data->xangle = 0;
		//data->zangle = 0;

		//data->light_x = -1;
		//data->light_y = 0;
		//data->light_color = 1;
		//data->light_intensity = 2;

		printf("Processing background image #%d: %s\n", pos, bgreader.imagename.c_str());
		
		TransformImage(imagename, data);

		data->scale = ((float)data->params.maxscale - 1.0F) * rand() / RAND_MAX + 1.0F;
		data->height = (int)(data->scale * data->params.winheight);
		data->width = (int)((data->scale * data->params.winwidth) * ((float)data->img_tran.cols / (float)data->img_tran.rows));
		data->x = (int)((0.1 + 0.8 * rand() / RAND_MAX) * (bgreader.image.cols - data->width));
		data->y = (int)((0.1 + 0.8 * rand() / RAND_MAX) * (bgreader.image.rows - data->height));

		char notes[300];
		sprintf(notes, "x = %d, y = %d, width = %d, height = %d, albedo = %f, Ka = %f, Kd = %f, Ks = %f, shin = %f, light_x = %f, light_y = %f, color = %f, intensity = %f",
			data->x, data->y, data->width, data->height,
			data->object_albedo, data->object_Ka, data->object_Kd, data->object_Ks, data->object_shininess,
			data->light_x, data->light_y, data->light_color, data->light_intensity);
		putText(bgreader.image, notes, Point(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 255, 255), 1);

		PlaceTransformedImage(bgreader.image, data);

		infowriter.WriteInfo(i, data);
		infowriter.WriteImage(i, data, bgreader.image);
	}
}