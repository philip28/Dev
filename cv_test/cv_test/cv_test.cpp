#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//#include <windows.h>

using namespace cv;

Mat ImgSrc, ImgGray;
Mat ImgEdges, ImgDst, ImgDstNorm, ImgDstNormScaled;
char src_name[] = "D:\\Dev\\cv_test\\x64\\Debug\\20170626_204837_40p.jpg";
char window_name[] = "cv_test";
int LowThreshold = 0, LowThresholdMax = 1000;
int ThresholdRatio = 3, KernelSize = 3;

/*
static void EdgeDetect(int, void*)
{
	blur(ImgGray, ImgEdges, Size(3, 3));
	Canny(ImgEdges, ImgEdges, LowThreshold, LowThreshold*ThresholdRatio, KernelSize);
	ImgDst = Scalar::all(0);
	ImgSrc.copyTo(ImgDst, ImgEdges);
	imshow(window_name, ImgDst);
}
*/

static void NormalizeTest()
{
//	int i = ImgSrc.type();
	ImgDst = Mat::zeros(ImgSrc.size(), ImgSrc.type());
//	int j = ImgDst.channels();
	normalize(ImgSrc, ImgDst, 100, 0, NORM_L2, CV_32FC1);
}

int main()
{
	int thresh = 200;

	ImgSrc = imread(src_name, IMREAD_COLOR);
	if (ImgSrc.empty())
	{
		return -1;
	}

	NormalizeTest();
/*
	cvtColor(ImgSrc, ImgGray, COLOR_BGR2GRAY);
	ImgDst = Mat::zeros(ImgSrc.size(), CV_32FC1);

	cornerHarris(ImgGray, ImgDst, 7, 5, 0.05, BORDER_DEFAULT);

	normalize(ImgDst, ImgDstNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(ImgDstNorm, ImgDstNormScaled);

	for (int j = 0; j < ImgDstNorm.rows; j++)
	{
		for (int i = 0; i < ImgDstNorm.cols; i++)
		{
			if ((int)ImgDstNorm.at<float>(j, i) > thresh)
			{
				circle(ImgDstNormScaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
*/

	namedWindow(window_name, WINDOW_AUTOSIZE);
	imshow("corners_window", ImgDst);


/*
	ImgDst.create(ImgSrc.size(), ImgSrc.type());

	cvtColor(ImgSrc, ImgSrcGray, COLOR_BGR2GRAY);

	namedWindow(window_name, WINDOW_AUTOSIZE);

	createTrackbar("Min Threshold:", window_name, &LowThreshold, LowThresholdMax, EdgeDetect);

	EdgeDetect(0, 0);
*/

	waitKey(0);

	return 0;
}