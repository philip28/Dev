#include "opencv2/opencv.hpp" 

using namespace cv;
using namespace std;

// Specifying minimum and maximum size parameters
#define MIN_LOGO_SIZE 100
#define MAX_LOGO_SIZE 300

int DetectVideoCapture(char *Cascade)
{
	// Load the Cascade Classifier Xml file
	CascadeClassifier LogoCascade(Cascade);

	// Create a VideoCapture object
	VideoCapture cap;

	// Check if camera opened successfully
	if (!cap.open(0)) return 0;

	Mat Frame, FrameBig, FrameGray;

	while (1) {

		// Reading each frame
		bool FrameRead = cap.read(FrameBig);

		// If frame not opened successfully
		if (!FrameRead)
			break;

		// Fixing the scaling factor
		float scale = 1200.0f / FrameBig.cols;

		// Resizing the image
		resize(FrameBig, Frame, Size(), scale, scale);

		// Converting to grayscale
		cvtColor(Frame, FrameGray, COLOR_BGR2GRAY);

		// Creating vector to store the detected logos' parameters
		vector<Rect> Logos;
		
		// Detect objects
		LogoCascade.detectMultiScale(FrameGray, Logos, 1.1, 5, 0, Size(MIN_LOGO_SIZE, MIN_LOGO_SIZE), Size(MAX_LOGO_SIZE, MAX_LOGO_SIZE));
		//		faceCascade.detectMultiScale(frameGray, faces);

		// Loop over each detected face
		for (int i = 0; i < Logos.size(); i++)
		{
			// Dimension parameters for bounding rectangle for face
			Rect LogoRect = Logos[i];

			// Drawing the bounding rectangle around the face
			rectangle(Frame, LogoRect, Scalar(128, 255, 0), 2);
		}

		// Display the resulting frame    
		imshow("Pepsi Detector", Frame);
		int k = waitKey(1);
		int n = 1;

		// Press ESC on keyboard to stop tracking
		if (k == 27)
			break;
		
		// Press Tab to save frame
		if (k == 9) {
			char iname[50];
			sprintf(iname, "c:\\dev\\frames\\%04d.jpg", n++);
			imwrite(iname, Frame);
			k = 0;
		}
	}
	// release the VideoCapture object
	cap.release();

	// Closes all the windows
	destroyAllWindows();

	return 0;
}

int DetectImage(char* SrcName, char* Cascade)
{
	Mat ImgSrcBig, ImgSrc, ImgGray;

	ImgSrcBig = imread(SrcName, IMREAD_COLOR);
	if (ImgSrcBig.empty())
	{
		return -1;
	}

	float scale = 1200.0f / ImgSrcBig.cols;
	resize(ImgSrcBig, ImgSrc, Size(), scale, scale);

	cvtColor(ImgSrc, ImgGray, COLOR_BGR2GRAY);

	vector<Rect> Logos;

	CascadeClassifier LogoCascade(Cascade);

	//LogoCascade.detectMultiScale(ImgGray, Logos, 1.1, 5, 0, Size(MIN_LOGO_SIZE, MIN_LOGO_SIZE), Size(MAX_LOGO_SIZE, MAX_LOGO_SIZE));
	LogoCascade.detectMultiScale(ImgGray, Logos);

	for (int i = 0; i < Logos.size(); i++)
	{
		Rect LogoRect = Logos[i];
		rectangle(ImgGray, LogoRect, Scalar(255, 255, 255), 2, LINE_4);
	}

	imshow("Pepsi Detector", ImgGray);
	waitKey(0);

	return 0;
}

int main(void)
{
	DetectVideoCapture("D:\\Dev\\PepsiDemo\\cascade\\cascade.xml");
//	DetectVideoCapture("D:\\Dev\\PepsiDemo\\cascade_10_0.999_0.1\\cascade.xml");
//	DetectImage("D:\\Dev\\PepsiDemo\\positive\\20170626_204830_sh_removal.jpg", "D:\\Dev\\PepsiDemo\\cascade\\cascade.xml");
//	DetectImage("c:\\dev\\frames\\0001.jpg", "D:\\Dev\\PepsiDemo\\cascade\\cascade.xml");
//	DetectImage("D:\\Dev\\PepsiDemo\\positive\\20170626_204830.jpg", "D:\\Dev\\PepsiDemo\\cascade_10_0.999_0.2_0h11m\\cascade.xml");
//	DetectImage("D:\\Dev\\PepsiDemo\\positive\\20170626_204830.jpg", "D:\\Dev\\PepsiDemo\\cascade_10_0.999_0.1_24_2h49m\\cascade.xml");
//	DetectImage("D:\\Dev\\PepsiDemo\\positive\\20170626_204830.jpg", "D:\\Dev\\PepsiDemo\\cascade_10_0.999_0.2_24_0h9m\\cascade.xml");
//	DetectImage("D:\\Dev\\PepsiDemo\\positive\\20170626_204830.jpg", "D:\\Dev\\PepsiDemo\\cascade\\cascade.xml");
//	DetectImage("D:\\Dev\\PepsiDemo\\positive\\20170706_174113_crop.jpg", "D:\\Dev\\PepsiDemo\\cascade\\cascade.xml");

	return 0;
}

