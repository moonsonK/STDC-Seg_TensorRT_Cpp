#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Segmenter.hpp"

int main(int argc, char** argv)
{
	std::string PathToEngineFile = "/home/integrity/Downloads/STDC1-Seg50.engine";
	Segmenter SegmentationObj(PathToEngineFile);

	SegmentationObj.LoadModel();
	SegmentationObj.AllocateMemory();
}