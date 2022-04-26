#ifndef __FILTERS_H__
#define __FILTERS_H__
//#define FILTERSIZE 3

void Conv2DSerial(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);
float get_std(float *inImg, int width, int height); 
void populate_blur_filter(double *outFilter, size_t filterEdgeLen, float stdev);
void GradientSobelSerial(float *inImg, float *sobelImg, float *gradientImg, int height, int width, size_t filterSIze); 
void ColorToGrayscaleSerial(float *input, float *output, unsigned int y, unsigned int x); 

#ifdef __CUDACC__
__global__ void ColorToGrayscale(float *inImg, float *outImg, int width, int height); 
__global__ void Conv2D(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize); 
__global__ void GradientSobel(float *inImg, float *sobelImg, float *gradientImg, int height, int width, size_t filterSize); 
__global__ void Conv2DTiled(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);
__global__ void GradientSobelTiled(float *inImg, float *sobelImg, float *gradientImg, int height, int width, size_t filterSize);
__global__ void Conv2DOpt(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);
__global__ void GradientSobelOpt(float *inImg, float *sobelImg, float *gradientImg, int height, int width, size_t filterSize);
__global__ void Conv2DOptRow(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);
__global__ void Conv2DOptCol(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize);

#endif

#endif // __FILTERS_H__
