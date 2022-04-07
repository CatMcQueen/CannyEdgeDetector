#include <wb.h>
#include "filters.cu"
#include "Otsus_Method.h"
#include "non_max_supp.h"

#define FILTERSIZE 3

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

int main(int argc, char *argv[])
{

	//////////////////////////////
	// Parameter Initialization //
	//////////////////////////////


	// Image parameters for wbLib
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;

	// Host side parameters
	float *hostInputImageData;
	float *hostGrayImageData;
	float *hostBlurImageData;
	float *hostGradientImageData;
	float *hostSobelImageData;

	// Device side parameters
	float *deviceInputImageData;
	float *deviceGrayImageData;
	float *deviceBlurImageData;
	float *deviceGradientImageData;
	float *deviceSobelImageData;
	
	// Filtering parameters
	float *BlurImageData;
	float *SobelImageData;
	float *NmsImageData;
	float *GradientImageData;

	// Otsu's Method parameters
	unsigned int *histogram;


	////////////////////
	// Image Handling //
	////////////////////


	// Parse the input arguments
	args = wbArg_read(argc, argv);

	// Read input file
	inputImageFile = wbArg_getInputFile(args, 0);

	// Import input image 
	inputImage = wbImport(inputImageFile);

	// Scrape info from input image
	imageWidth  = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	
	// Define new output image
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	// Define output image data
	hostInputImageData = wbImage_getData(inputImage);

	// CHANGE THIS TO CHANGE OUTPUT IMAGE
	BlurImageData = wbImage_getData(outputImage);


	////////////////////////////////
	// Host Memory Initialization //
	////////////////////////////////


	// Allocate memory on host
	hostGrayImageData     = (float *)malloc(imageHeight*imageWidth*sizeof(float));
	hostBlurImageData     = (float *)malloc(imageHeight*imageWidth*sizeof(float));
	hostSobelImageData    = (float *)malloc(imageHeight*imageWidth*sizeof(float));
	hostGradientImageData = (float *)malloc(imageHeight*imageWidth*sizeof(float));

	// Allocate memory for serial filtering
	//BlurImageData     = (float *)malloc(imageHeight*imageWidth*sizeof(float));
	SobelImageData    = (float *)malloc(imageHeight*imageWidth*sizeof(float));
	GradientImageData = (float *)malloc(imageHeight*imageWidth*sizeof(float));

	// Allocate memory on host and set to 0
	histogram = (unsigned int *)calloc(256, sizeof(unsigned int));


	/////////////////////////
	// Image Preprocessing //
	/////////////////////////


	// Create filter skeleton
	double filter[FILTERSIZE][FILTERSIZE];

	// Fill the gaussian filter
	populate_blur_filter(filter);

	// ?????
	int filterSize = (int)FILTERSIZE;


	//////////////////////////////////
	// Device Memory Initialization //
	//////////////////////////////////


	// Start total program timer
	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	// Start memory allocation timer
	wbTime_start(GPU, "Doing GPU memory allocation");

	// Allocate memory on device
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceGrayImageData, imageWidth*imageHeight*sizeof(float));
	cudaMalloc((void **)&deviceBlurImageData, imageWidth*imageHeight*sizeof(int));
	cudaMalloc((void **)&deviceSobelImageData, imageWidth*imageHeight*sizeof(int));
	cudaMalloc((void **)&deviceGradientImageData, imageWidth*imageHeight*sizeof(int));

	// Stop memory allocation timer
	wbTime_stop(GPU, "Doing GPU memory allocation");

	// Start memory copy timer
	wbTime_start(Copy, "Copying data to the GPU");

	// Copy input image from host to device
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageChannels*imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);


	///////////////////
	// GPU Execution //
	///////////////////


	// Start computation timer
	wbTime_start(Compute, "Doing the computation on the GPU");

	// Number of threads/block is 16
	int blocksize = 16;

	// Initialize x and y block dimension to blocksize
	dim3 BlockDim(blocksize,blocksize);

	// Set x and y grid dimension 
	dim3 GridDim(((imageWidth+BlockDim.x-1)/BlockDim.x), ((imageHeight+BlockDim.y-1)/BlockDim.y));  

	// Call RGB to grayscale conversion kernel
	ColorToGrayscale<<<GridDim, BlockDim>>>(deviceInputImageData, deviceGrayImageData, imageWidth, imageHeight);

	// Call image burring kernel
	//Conv2D<<<GridDim, BlockDim>>>(deviceGrayImageData, deviceBlurImageData, filter, imageWidth, imageHeight, filterSize);

	// Call sobel filtering kernel
	//GradientSobel<<<GridDim, BlockDim>>>(deviceBlurImageData, deviceSobelImageData, deviceSobelImageData, imageHeight, imageWidth); 

	// Stop computation timer
	wbTime_stop(Compute, "Doing the computation on the GPU");


	////////////////////
	// Device Results //
	////////////////////


	// Start device memory copy timer
	wbTime_start(Copy, "Copying data from the GPU");

	// Copy data from device back to host
	cudaMemcpy(hostGrayImageData, deviceGrayImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(hostBlurImageData, deviceBlurImageData, imageWidth*imageHeight*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(hostGrayImageData, deviceGrayImageData, imageWidth*imageHeight*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(hostSobelImageData, deviceSobelImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(hostGradientImageData, deviceGradientImageData, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice); 

	// Stop memory timer
	wbTime_stop(Copy, "Copying data from the GPU");

	// Stop total program timer
	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	
	////////////////////
	// Host Execution //
	////////////////////

	// Blur image using Gaussian Kernel
	Conv2DSerial(hostGrayImageData, BlurImageData, filter, imageWidth, imageHeight, filterSize);

	// Calculate gradient using Sobel Operators
	GradientSobelSerial(BlurImageData, SobelImageData, GradientImageData, imageHeight, imageWidth);

  // Suppress non-maximum pixels along gradient
  nms(SobelImageData, NmsImageData, GradientImageData, imageHeight, imageWidth);

	// Calculate histogram of blurred image
	Histogram_Sequential(BlurImageData, histogram, imageWidth, imageHeight);

	// Calculate threshold using Otsu's Method
	double thresh = Otsu_Sequential(histogram);


	////////////////////
	// Debugging Info //
	////////////////////


	// Print info
	printf("\n");
	printf("Width = %u\n",imageWidth);
	printf("Height = %u\n",imageHeight);
	printf("InputImage[0] = %f\n",hostInputImageData[0]);
	printf("Histogram[0] = %u\n",histogram[0]);
	printf("Histogram[1] = %u\n",histogram[1]);
	printf("Histogram[20] = %u\n",histogram[20]);
	printf("Histogram[45] = %u\n",histogram[45]);
	printf("Histogram[56] = %u\n",histogram[56]);
	printf("Image[0] = %f\n",hostGrayImageData[0]);
	printf("Image[1] = %f\n",hostGrayImageData[1]);
	printf("Image[36] = %f\n",hostGrayImageData[36]);
	printf("Image[400] = %f\n",hostGrayImageData[400]);
	printf("Image[900] = %f\n",hostGrayImageData[900]);
	printf("Image[1405] = %f\n",hostGrayImageData[1405]);
	printf("Image[85000] = %f\n",hostGrayImageData[85000]);
	printf("First row of Gaussian filter = %f %f %f\n",filter[0][0], filter[0][1], filter[0][2]);
	printf("Second row of Gaussian filter = %f %f %f\n",filter[1][0], filter[1][1], filter[1][2]);
	printf("Third row of Gaussian filter = %f %f %f\n",filter[2][0], filter[2][1], filter[2][2]);
	printf("Otsu's Threshold = %f\n", thresh);
	printf("\n");

	// Export image
	char *oFile = wbArg_getOutputFile(args);
	wbExport(oFile, outputImage);


	//////////////
	// Clean Up //
	//////////////


	// Destory all cuda memory
	cudaFree(deviceInputImageData);
	cudaFree(deviceGrayImageData);
	cudaFree(deviceBlurImageData);
	cudaFree(deviceSobelImageData);
	cudaFree(deviceGradientImageData);

	// Destroy all host memory
	free(hostBlurImageData);
	free(hostSobelImageData);
	free(hostGradientImageData);
	//free(BlurImageData);
	free(SobelImageData);
	free(NmsImageData);
	free(GradientImageData);
	free(histogram);

	// Destroy images
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
