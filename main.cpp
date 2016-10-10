#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "hsv_gpu.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	//int rhue1=110, rhue2=180, rsaturation1=133, rsaturation2=255, rvalue1=25, rvalue2=255;
	//int yhue1=79, yhue2=93, ysaturation1=119, ysaturation2=255, yvalue1=30, yvalue2=255;
	//int ghue1=40, ghue2=70, gsaturation1=89, gsaturation2=255, gvalue1=33, gvalue2=255;


	Mat src, hsv_dst, h_dst, s_dst, v_dst;
	HSV rHSV;
	rHSV.setThreshold(110, 180, 133, 255, 25, 255);

	VideoCapture cp(0);
	
	cp.read(src);
	src.copyTo(hsv_dst);
	src.copyTo(h_dst);
	src.copyTo(s_dst);
	src.copyTo(v_dst);

	while(1)
	{
		cp.read(src);

		rHSV.adjustHSV(&src, &hsv_dst);
		rHSV.adjustH(&src, &h_dst);
		rHSV.adjustS(&src, &s_dst);
		rHSV.adjustV(&src, &v_dst);


		imshow("src", src);
		imshow("hsv_dst", hsv_dst);
		imshow("h_dst", h_dst);
		imshow("s_dst", s_dst);
		imshow("v_dst", v_dst);

		if(waitKey(30) == 27) break;
	}
	//waitKey(0);

	return 0;
}
