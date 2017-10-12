#define _CRT_SECURE_NO_WARNINGS

#include "utility.h"
#include "BackgroundImageReader.h"
#include "InfoWriter.h"

bool WrapCylinderTransform(Mat src, Mat &dst, double r, double phi, unsigned int bgcolor)
{
	double cx, cxp, xp, dcx;
	double alpha, gamma;
	int minx, maxx;
	int y, l, inth;
	int dst_width, dst_height;
	
	if (src.depth() != CV_8U || src.channels() != 1 || src.dims != 2)
	{
		CV_Error(CV_StsBadArg, "Source must be a two-dimensional array of CV_8UC1 type.");
		return false;
	}

	inth = src.cols / 2;
	cx = (double)src.cols / 2; // projection of the central point on X axis
	cxp = r * sin(phi); // central x after rotation
	dcx = cx - cxp;

	// Border points, depending on parity of cols
	alpha = atan2(inth, r);
	gamma = alpha + (PI / 2 - phi);
	xp = r * cos(gamma);
	minx = (int)(xp + dcx);

	if (src.cols % 2)
		alpha = atan2(inth, r);
	else
		alpha = atan2(inth-1, r);
	gamma = (PI / 2 - phi) - alpha;
	xp = r * cos(gamma);
	maxx = (int)(xp + dcx);

	// initialize dest image
	dst_width = maxx - minx + 1;
	dst_height = src.rows;
	dst.create(dst_height, dst_width, CV_8UC1);
	dst.setTo(Scalar(bgcolor));

	for (y = 0; y < dst_height; y++)
	{
		dst.at<unsigned char>(y, 0) = src.at<unsigned char>(y, (int)(cx - inth));
		if (src.cols % 2)
			dst.at<unsigned char>(y, dst_width-1) = src.at<unsigned char>(y, (int)(cx + inth));

		for (l = inth-1; l >= 1; l--)
		{
			alpha = atan2(l, r);
			gamma = alpha + (PI / 2 - phi); // "left" of new central point
			xp = r * cos(gamma);
			dst.at<unsigned char>(y, (int)(xp + dcx) - minx) = src.at<unsigned char>(y, (int)(cx - l));
			gamma = (PI / 2 - phi) - alpha; // "right" of the new central point 
			xp = r * cos(gamma);
			dst.at<unsigned char>(y, (int)(xp + dcx) - minx) = src.at<unsigned char>(y, (int)(cx + l));
		}

		// Central point
		gamma = PI / 2 - phi;
		xp = r * cos(gamma);
		dst.at<unsigned char>(y, (int)(xp + dcx) - minx) = src.at<unsigned char>(y, (int)(cx));
	}

	return true;
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

bool TransformImage(string imagename, ImageTransformData* data)
{
	Mat img_erode, img_dilate;
	unsigned char *pmask, *psrc, *perode, *pdilate;
	unsigned char dd, de;

	data->src = imread(imagename.c_str(), IMREAD_GRAYSCALE);
	if (data->src.empty())
		CV_Error(CV_StsBadArg, "Error opening the source image");

	data->mask = data->src.clone();
	img_erode = data->src.clone();
	img_dilate = data->src.clone();

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

	erode(data->src, img_erode, Mat(), Point(1,1), 1, BORDER_REPLICATE);
	dilate(data->src, img_dilate, Mat(), Point(1, 1), 1, BORDER_REPLICATE);

	for (int row = 0; row < data->mask.rows; row++)
	{
		for (int col = 0; col < data->mask.cols; col++)
		{
			pmask = &data->mask.at<unsigned char>(row, col);
			if ((*pmask) == 0)
			{
				psrc = &data->src.at<unsigned char>(row, col); 
				perode = &img_erode.at<unsigned char>(row, col);
				pdilate = &img_dilate.at<unsigned char>(row, col);
				de = data->params.bgcolor - *perode;
				dd = *pdilate - data->params.bgcolor;
				if (de >= dd && de > data->params.bgthreshold)
				{
					*psrc = *perode;
				}
				if (dd > de && dd > data->params.bgthreshold)
				{
					*psrc = *pdilate;
				}
			}
		}
	}

	Mat img_wrap, mask_wrap;

	data->r = data->r * data->src.cols / PI;
	WrapCylinderTransform(data->src, img_wrap, data->r, data->phi, data->params.bgcolor);
	WrapCylinderTransform(data->mask, mask_wrap, data->r, data->phi, 0);

	GaussianBlur(mask_wrap, mask_wrap, Size(3, 3), 0, 0);

	data->trans_img = img_wrap.clone();
	data->trans_mask = mask_wrap.clone();
	data->tr_width = img_wrap.cols;
	data->tr_height = img_wrap.rows;

//	ViewAngleTransform(img_wrap, data->trans_img, 0, PI/8, 0, MAX(data->src.cols, data->src.rows)*2);
//	Mat ti;
//	ti.create(data->src.rows, data->src.cols, CV_8UC1);
//	ti.setTo(Scalar(bgcolor));
//	ViewAngleTransform(data->trans_img, ti, 0, PI/4, 0, MAX(data->src.cols, data->src.rows) * 2);

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

	resize(data->trans_img, img, img.size());
	resize(data->trans_mask, mask, mask.size());
//	imshow("data->trans_img", data->trans_img);
//	waitKey(0);

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
//	imshow("background", background);
//	waitKey(0);

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
	
	for (i = 0; i < bgreader.count; i++)
	{
		if (data->params.random)
			pos = bgreader.NextRandom();
		else
			pos = bgreader.Next();

		if (data->params.maxscale < 0.0)
		{
			data->params.maxscale = MIN(0.7F * bgreader.image.cols / data->params.winwidth,
				0.7F * bgreader.image.rows / data->params.winheight);
		}
		if (data->params.maxscale < 1.0F)
			CV_Error(CV_StsBadArg, "Scaling down not supported\n");

		data->r = data->params.minrad + ((double)rand() / RAND_MAX) * (data->params.maxrad - data->params.minrad);
		data->phi = data->params.maxcylrot * (1.0 - 2.0 * rand() / RAND_MAX);
		data->hangle = data->params.maxhangle * rand() / RAND_MAX;
		data->vangle = data->params.maxvangle * rand() / RAND_MAX;

		printf("Processing background image #%d: %s\n", pos, bgreader.imagename.c_str());

		TransformImage(imagename, data);

		data->scale = ((float)data->params.maxscale - 1.0F) * rand() / RAND_MAX + 1.0F;
		data->height = (int)(data->scale * data->params.winheight);
		data->width = (int)((data->scale * data->params.winwidth) * ((float)data->tr_width / (float)data->tr_height));
		data->x = (int)((0.1 + 0.8 * rand() / RAND_MAX) * (bgreader.image.cols - data->width));
		data->y = (int)((0.1 + 0.8 * rand() / RAND_MAX) * (bgreader.image.rows - data->height));

		PlaceTransformedImage(bgreader.image, data);

		infowriter.WriteInfo(i, data);
		infowriter.WriteImage(i, data, bgreader.image);
	}
}