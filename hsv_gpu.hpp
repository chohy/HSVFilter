#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class HSV
{
	public:
		//HSV();
		void setThreshold(double h1, double h2, double s1, double s2, double v1, double v2);
		void adjustHSV(Mat* src, Mat* dst);
		void adjustH(Mat* src, Mat* dst);
		void adjustS(Mat* src, Mat* dst);
		void adjustV(Mat* src, Mat* dst);
	
	private:
		//hsv filter threshold variable
		double hue1, hue2, saturation1, saturation2, value1, value2;
	
		Mat hsrc;
		uchar *dev_src, *dev_hsrc;
	
		gpu::GpuMat gsrc;
		gpu::GpuMat gdst;		
};
