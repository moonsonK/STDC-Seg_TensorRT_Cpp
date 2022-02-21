#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "Segmenter.hpp"

int main(int argc, char** argv)
{
	std::string PathToEngineFile = "/home/integrity/Downloads/STDC2-Seg75Opset11.engine";

	Segmenter SegmentationObj(PathToEngineFile);

	bool InitSuccessful = SegmentationObj.LoadAndPrepareModel();

	cv::Mat Frame = cv::imread("/home/integrity/Downloads/TestImg.png", cv::IMREAD_UNCHANGED);
	// Frame = Frame[:,:,0:3];

	std::vector<cv::Mat> Masks;
	SegmentationObj.ProcessFrame(Frame, Masks);

    // cv::imwrite("/home/integrity/Downloads/Test.jpeg", *(OutputMasks+8)*255);
	// std::cout << OutputMasks << std::endl;

	cv::Mat OutputImage = SegmentationObj.DrawMasks(Masks);

	cv::imwrite("/home/integrity/Downloads/Test.jpeg", OutputImage);

}