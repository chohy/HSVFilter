#include "hsv_gpu.hpp"

void __global__ hsv_filter(uchar *dev_src, uchar *dev_hsrc, int cols,
			   int hue1, int hue2, int saturation1, int saturation2, int value1, int value2)
{
	int tid = blockIdx.x;
	int c;

	for(c=0;c<cols;c++)
	{
		if(dev_hsrc[cols*3*tid + 3*c + 0] < hue1 || dev_hsrc[cols*3*tid + 3*c + 0] > hue2 ||
		   dev_hsrc[cols*3*tid + 3*c + 1] < saturation1 || dev_hsrc[cols*3*tid + 3*c + 1] > saturation2 ||
		   dev_hsrc[cols*3*tid + 3*c + 2] < value1 || dev_hsrc[cols*3*tid + 3*c + 1] > value2)
		{
			dev_src[cols*3*tid + 3*c + 0] = 0;
			dev_src[cols*3*tid + 3*c + 1] = 0;
			dev_src[cols*3*tid + 3*c + 2] = 0;
		}
	}
}

void __global__ h_filter(uchar *dev_src, uchar *dev_hsrc, int cols, int hue1, int hue2)
{
	int tid = blockIdx.x;
	int c;

	for(c=0;c<cols;c++)
	{
		if(dev_hsrc[cols*3*tid + 3*c + 0] < hue1 || dev_hsrc[cols*3*tid + 3*c + 0] > hue2)
		{
			dev_src[cols*3*tid + 3*c + 0] = 0;
			dev_src[cols*3*tid + 3*c + 1] = 0;
			dev_src[cols*3*tid + 3*c + 2] = 0;
		}
	}
}

void __global__ s_filter(uchar *dev_src, uchar *dev_hsrc, int cols, int saturation1, int saturation2)
{
	int tid = blockIdx.x;
	int c;

	for(c=0;c<cols;c++)
	{
		if(dev_hsrc[cols*3*tid + 3*c + 1] < saturation1 || dev_hsrc[cols*3*tid + 3*c + 1] > saturation2)
		{
			dev_src[cols*3*tid + 3*c + 0] = 0;
			dev_src[cols*3*tid + 3*c + 1] = 0;
			dev_src[cols*3*tid + 3*c + 2] = 0;
		}
	}
}

void __global__ v_filter(uchar *dev_src, uchar *dev_hsrc, int cols, int value1, int value2)
{
	int tid = blockIdx.x;
	int c;

	for(c=0;c<cols;c++)
	{
		if(dev_hsrc[cols*3*tid + 3*c + 2] < value1 || dev_hsrc[cols*3*tid + 3*c + 2] > value2)
		{
			dev_src[cols*3*tid + 3*c + 0] = 0;
			dev_src[cols*3*tid + 3*c + 1] = 0;
			dev_src[cols*3*tid + 3*c + 2] = 0;
		}
	}
}

void HSV::setThreshold(double h1, double h2, double s1, double s2, double v1, double v2)
{
	hue1 = h1;
	hue2 = h2;
	saturation1 = s1;
	saturation2 = s2;
	value1 = v1;
	value2 = v2;
}

void HSV::adjustHSV(Mat* src, Mat* dst)
{
	gsrc.upload(*src);
	gpu::cvtColor(gsrc, gdst, CV_RGB2HSV);
	gdst.download(hsrc);

	cudaMalloc((void**)&dev_src, src->rows*src->cols*src->channels()*src->elemSize1());
	cudaMalloc((void**)&dev_hsrc, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1());
	cudaMemcpy(dev_src, src->data, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hsrc, hsrc.data, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1(), cudaMemcpyHostToDevice);

	hsv_filter<<<src->rows, 1>>>(dev_src, dev_hsrc, src->cols, hue1, hue2, saturation1, saturation2, value1, value2);

	cudaMemcpy(dst->data, dev_src, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_src);
	cudaFree(dev_hsrc);
	
}

void HSV::adjustH(Mat* src, Mat* dst)
{
	gsrc.upload(*src);
	gpu::cvtColor(gsrc, gdst, CV_RGB2HSV);
	gdst.download(hsrc);

	cudaMalloc((void**)&dev_src, src->rows*src->cols*src->channels()*src->elemSize1());
	cudaMalloc((void**)&dev_hsrc, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1());
	cudaMemcpy(dev_src, src->data, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hsrc, hsrc.data, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1(), cudaMemcpyHostToDevice);

	h_filter<<<src->rows, 1>>>(dev_src, dev_hsrc, src->cols, hue1, hue2);

	cudaMemcpy(dst->data, dev_src, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_src);
	cudaFree(dev_hsrc);
}

void HSV::adjustS(Mat* src, Mat* dst)
{
	gsrc.upload(*src);
	gpu::cvtColor(gsrc, gdst, CV_RGB2HSV);
	gdst.download(hsrc);

	cudaMalloc((void**)&dev_src, src->rows*src->cols*src->channels()*src->elemSize1());
	cudaMalloc((void**)&dev_hsrc, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1());
	cudaMemcpy(dev_src, src->data, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hsrc, hsrc.data, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1(), cudaMemcpyHostToDevice);

	s_filter<<<src->rows, 1>>>(dev_src, dev_hsrc, src->cols, saturation1, saturation2);

	cudaMemcpy(dst->data, dev_src, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_src);
	cudaFree(dev_hsrc);
}

void HSV::adjustV(Mat* src, Mat* dst)
{
	gsrc.upload(*src);
	gpu::cvtColor(gsrc, gdst, CV_RGB2HSV);
	gdst.download(hsrc);

	cudaMalloc((void**)&dev_src, src->rows*src->cols*src->channels()*src->elemSize1());
	cudaMalloc((void**)&dev_hsrc, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1());
	cudaMemcpy(dev_src, src->data, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hsrc, hsrc.data, hsrc.rows*hsrc.cols*hsrc.channels()*hsrc.elemSize1(), cudaMemcpyHostToDevice);

	v_filter<<<src->rows, 1>>>(dev_src, dev_hsrc, src->cols, value1, value2);

	cudaMemcpy(dst->data, dev_src, src->rows*src->cols*src->channels()*src->elemSize1(), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_src);
	cudaFree(dev_hsrc);
}
