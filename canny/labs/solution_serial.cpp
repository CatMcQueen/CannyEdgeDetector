#include <stdio.h>
#include <wb.h>
#include "Otsus_Method.h"
#include "filters.h"
#include "Edge_Connection.h"
#include "non_max_supp.h"

// Use for bypassing phases for testing and debug printing
//#include "test-code.h"

// Also modify the main function to launch thekernel.
int main(int argc, char *argv[]) {

    //////////////////////////////
    // Parameter Initialization //
    //////////////////////////////


    // Image parameters for wbLib
    wbArg_t args;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    size_t filterSize = 3; // default value
    char *inputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;

    float *hostInputImageData;

    // Filtering parameters
    float *BlurImageData;
    float *GrayImageData;
    float *GradMagData;
    float *GradPhaseData;
    float *NmsImageData;

    // Otsu's Method parameters
    unsigned int *histogram;

    // Edge Connection parameters
    float *weakEdgeImage;
    float *edgeImage;


    ////////////////////
    // Image Handling //
    ////////////////////


    // Parse the input arguments
    args = wbArg_read(argc, argv);

    // Read input file
    inputImageFile = wbArg_getInputFile(args, 0);
    filterSize = wbArg_getInputFilterSize(args);

    // Import input image 
    inputImage = wbImport(inputImageFile);

    // Scrape info from input image
    imageWidth  = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    // Define new output image
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    // Define output image data
    hostInputImageData = wbImage_getData(inputImage);

    ////////////////////////////////
    // Host Memory Initialization //
    ////////////////////////////////

    // Start total program timer (in nanoseconds)
    wbTime_start(GPU, "Doing Computation (memory + compute)");

    // Start memory allocation timer
    wbTime_start(GPU, "Doing memory allocation");

    // Fill the gaussian filter
    double *filter    = (double *)calloc(filterSize*filterSize, sizeof(double));
    populate_blur_filter(filter, filterSize);

	// Allocate memory on host
  GrayImageData     = (float *)calloc(imageHeight*imageWidth, sizeof(float));
	BlurImageData     = (float *)calloc(imageHeight*imageWidth, sizeof(float));
	GradMagData    		= (float *)calloc(imageHeight*imageWidth, sizeof(float));
	GradPhaseData 		= (float *)calloc(imageHeight*imageWidth, sizeof(float));
	NmsImageData      = (float *)calloc(imageHeight*imageWidth, sizeof(float));
	weakEdgeImage     = (float *)calloc(imageHeight*imageWidth, sizeof(float));
	edgeImage         = (float *)calloc(imageHeight*imageWidth, sizeof(float));


  // Initialize memory for the output image
  // Note - input image is 3 channels. Other phases only have 1 channel
  float *outData = (float *)calloc(imageHeight*imageWidth,sizeof(float));
	outputImage = wbImage_new(imageWidth, imageHeight, 1, outData);
  
	histogram = (unsigned int *)calloc(256, sizeof(unsigned int));

    // Stop memory allocation timer
    wbTime_stop(GPU, "Doing memory allocation");

	/////////////////////////
	// Image Preprocessing //
	/////////////////////////


	////////////////////
	// Host Execution //
	////////////////////

	// Start computation timer
	wbTime_start(Compute, "Doing the computation");

  // GrayImageData Serial
    wbTime_start(Compute, "ColorToGrayscale computation");
  ColorToGrayscaleSerial(hostInputImageData, GrayImageData, imageWidth, imageHeight);
    wbTime_stop(Compute, "ColorToGrayscale computation");

	// Blur image using Gaussian Kernel
    wbTime_start(Compute, "Conv2D computation");
	Conv2DSerial(GrayImageData, BlurImageData, filter, imageWidth, imageHeight, filterSize);

	// Calculate gradient using Sobel Operators
    wbTime_start(Compute, "GradientSobelS computation");
	GradientSobelSerial(BlurImageData, GradMagData, GradPhaseData, imageHeight, imageWidth, filterSize);

  // Suppress non-maximum pixels along gradient
    wbTime_start(Compute, "Non-maximum Suppression computation");
  nms(GradMagData, NmsImageData, GradPhaseData, imageHeight, imageWidth);
    wbTime_stop(Compute, "Non-maximum Suppression computation");

	// Calculate histogram of blurred image
    wbTime_start(Compute, "Histogram computation");
	Histogram_Sequential(NmsImageData, histogram, imageWidth, imageHeight);
    wbTime_stop(Compute, "Histogram computation");

    // Calculate threshold using Otsu's Method
    wbTime_start(Compute, "Otsu's computation");
    double thresh = Otsu_Sequential(histogram, imageWidth, imageHeight);
    wbTime_stop(Compute, "Otsu's computation");

    // Calculate strong, weak, and non edges using thresholds
    wbTime_start(Compute, "Threshold Detection computation");
    threshold_detection_serial(NmsImageData, weakEdgeImage, edgeImage, thresh, imageWidth, imageHeight);
    wbTime_stop(Compute, "Threshold Detection computation");

    // Connect edges by connecting weak edges to strong edges
    wbTime_start(Compute, "Edge connection computation");
    edge_connection_serial(weakEdgeImage, edgeImage, imageWidth, imageHeight);
    wbTime_stop(Compute, "Edge connection computation");

	// Stop computation timer
	wbTime_stop(Compute, "Doing the computation");

  // Copy image data for output image (choose 1 - can only log one at a time for now
  //memcpy(outData, GrayImageData, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, BlurImageData, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, GradMagData, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, GradPhaseData, imageHeight*imageWidth*sizeof(float));
  memcpy(outData, NmsImageData, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, weakEdgeImage, imageHeight*imageWidth*sizeof(float));
  //memcpy(outData, edgeImage, imageHeight*imageWidth*sizeof(float));

#if (PRINT_DEBUG)
  //FILE *testThin = fopen("nmsThin.txt", "w");
  //for(int x = 0; x < imageWidth; ++x)
  //{
  //  for(int y = 0; y < imageHeight; ++y)
  //  {
  //    fprintf(testThin, "%f ", NmsImageData[x + y*imageWidth]);
  //  }
  //  fprintf(testThin, "\n");
  //}
  //fclose(testThin);
  //testThin = 0;

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
	printf("Image[0] = %f\n",GrayImageData[0]);
	printf("Image[1] = %f\n",GrayImageData[1]);
	printf("Image[36] = %f\n",GrayImageData[36]);
	printf("Image[400] = %f\n",GrayImageData[400]);
	printf("Image[900] = %f\n",GrayImageData[900]);
	printf("Image[1405] = %f\n",GrayImageData[1405]);
	printf("Image[85000] = %f\n",GrayImageData[85000]);
	//printf("First row of Gaussian filter = %f %f %f\n",filter[0], filter[1], filter[2]);
	//printf("Second row of Gaussian filter = %f %f %f\n",filter[0 + 1*filterSize], filter[1 + 1*filterSize], filter[2 + 1*filterSize]);
	//printf("Third row of Gaussian filter = %f %f %f\n",filter[2*filterSize], filter[1 + 2*filterSize], filter[2 + 2*filterSize]);
  for(size_t row = 0; row < filterSize; ++row)
  {
    printf("Row=%ld of Gaussian filter = ",row);
    for(size_t col = 0; col < filterSize; ++col)
    {
            printf("%f ", filter[col + filterSize*row]);
        }
        printf("\n");
    }
    printf("Otsu's Threshold = %f\n", thresh);
    printf("\n");
#endif

    // Export image
    char *oFile = wbArg_getOutputFile(args);
    wbExport(oFile, outputImage);


    //////////////
    // Clean Up //
    //////////////


    // Destroy all host memory
    free(GradMagData);
    free(GradPhaseData);
    free(NmsImageData);
    free(GrayImageData);
    free(BlurImageData);
    free(histogram);
    free(weakEdgeImage);
    free(edgeImage);

    // Destroy images
    wbImage_delete(outputImage); // Handles free of outData
    wbImage_delete(inputImage);

    return 0;
}

