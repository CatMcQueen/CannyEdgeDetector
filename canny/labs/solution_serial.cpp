#include <stdio.h>
#include <wb.h>
#include <omp.h>
#include "Otsus_Method.h"
#include "filters.h"
#include "Edge_Connection.h"
#include "non_max_supp.h"

// Use for bypassing phases for testing and debug printing
//#include "test-code.h"

// Also modify the main function to launch thekernel.
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
    float stdev;
    size_t filterSize = 3; // default value
    char *inputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;

    // Filtering parameters
    float *hostInputImageData;
    float *BlurImageData;
    float *GrayImageData;
    float *GradMagData;
    float *GradPhaseData;
    float *NmsImageData;
    float *weakEdgeImage;
    float *edgeImage;

    // Otsu's Method parameters
    unsigned int *histogram;


    ////////////////////
    // Image Handling //
    ////////////////////


    // Parse the input arguments
    args = wbArg_read(argc, argv);

    // Read input file
    inputImageFile = wbArg_getInputFile(args, 0);
    stdev  = wbArg_getInputFilterSize(args);

    // Import input image
    inputImage = wbImport(inputImageFile);

    // Scrape info from input image
    imageWidth = wbImage_getWidth(inputImage);
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

    // Allocate memory on host
    GrayImageData = (float *)calloc(imageHeight * imageWidth, sizeof(float));
    BlurImageData = (float *)calloc(imageHeight * imageWidth, sizeof(float));
    GradMagData = (float *)calloc(imageHeight * imageWidth, sizeof(float));
    GradPhaseData = (float *)calloc(imageHeight * imageWidth, sizeof(float));
    NmsImageData = (float *)calloc(imageHeight * imageWidth, sizeof(float));
    weakEdgeImage = (float *)calloc(imageHeight * imageWidth, sizeof(float));
    edgeImage = (float *)calloc(imageHeight * imageWidth, sizeof(float));

    // Initialize memory for the output image
    // Note - input image is 3 channels. Other phases only have 1 channel
    float *outData = (float *)calloc(imageHeight * imageWidth, sizeof(float));
    outputImage = wbImage_new(imageWidth, imageHeight, 1, outData);

    histogram = (unsigned int *)calloc(256, sizeof(unsigned int));

    // Stop memory allocation timer
    wbTime_stop(GPU, "Doing memory allocation");


    ////////////////////
    // Host Execution //
    ////////////////////


    // Start computation timer
    wbTime_start(Compute, "Doing the computation");

    // GrayImageData Serial
    wbTime_start(Compute, "ColorToGrayscale computation");
        ColorToGrayscaleSerial(hostInputImageData, GrayImageData, imageWidth, imageHeight);
    wbTime_stop(Compute, "ColorToGrayscale computation");

    // solve for stdev of center row
    stdev = get_std(GrayImageData, imageWidth, imageHeight);
    filterSize = 2*ceil(stdev*3)+1;
    printf("\n");
    printf("Stdev = %f\n", stdev);
    // Fill the gaussian filter
    double *filter = (double *)calloc(filterSize * filterSize, sizeof(double));
    populate_blur_filter(filter, filterSize, stdev);
    

    // Blur image using Gaussian Kernel
    wbTime_start(Compute, "Conv2D computation");
        Conv2DSerial(GrayImageData, BlurImageData, filter, imageWidth, imageHeight, filterSize);
    wbTime_stop(Compute, "Conv2D computation");

    // Calculate gradient using Sobel Operators
    wbTime_start(Compute, "GradientSobelS computation");
        GradientSobelSerial(BlurImageData, GradMagData, GradPhaseData, imageHeight, imageWidth, filterSize);
    wbTime_stop(Compute, "GradientSobelS computation");

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
        //double thresh = Otsu_Sequential_Optimized(histogram, imageWidth, imageHeight);
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


    ////////////////////////
    // Logging and Output //
    ////////////////////////


    // Copy image data for output image (choose 1 - can only log one at a time for now
    // memcpy(outData, GrayImageData, imageHeight*imageWidth*sizeof(float));
    // memcpy(outData, BlurImageData, imageHeight*imageWidth*sizeof(float));
    // memcpy(outData, GradMagData, imageHeight*imageWidth*sizeof(float));
    // memcpy(outData, GradPhaseData, imageHeight*imageWidth*sizeof(float));
    memcpy(outData, NmsImageData, imageHeight * imageWidth * sizeof(float));
    // memcpy(outData, weakEdgeImage, imageHeight*imageWidth*sizeof(float));
    // memcpy(outData, edgeImage, imageHeight*imageWidth*sizeof(float));

    // Export image
    char *oFile = wbArg_getOutputFile(args);
    wbExport(oFile, outputImage);


    ////////////////////
    // Debugging Info //
    ////////////////////


    #if (PRINT_DEBUG)
        // FILE *testThin = fopen("nmsThin.txt", "w");
        // for(int x = 0; x < imageWidth; ++x)
        //{
        //  for(int y = 0; y < imageHeight; ++y)
        //  {
        //    fprintf(testThin, "%f ", NmsImageData[x + y*imageWidth]);
        //  }
        //  fprintf(testThin, "\n");
        //}
        // fclose(testThin);
        // testThin = 0;

        // Print info
        printf("\n");
        printf("Width = %u\n", imageWidth);
        printf("Height = %u\n", imageHeight);
        printf("InputImage[0] = %f\n", hostInputImageData[0]);
        printf("Histogram[0] = %u\n", histogram[0]);
        printf("Histogram[1] = %u\n", histogram[1]);
        printf("Histogram[20] = %u\n", histogram[20]);
        printf("Histogram[49] = %u\n", histogram[49]);
        printf("Histogram[56] = %u\n", histogram[56]);
        printf("Histogram[255] = %u\n", histogram[255]);
        printf("Image[0] = %f\n", GrayImageData[0]);
        printf("Image[1] = %f\n", GrayImageData[1]);
        printf("Image[36] = %f\n", GrayImageData[36]);
        printf("Image[400] = %f\n", GrayImageData[400]);
        printf("Image[900] = %f\n", GrayImageData[900]);
        printf("Image[1405] = %f\n", GrayImageData[1405]);
        printf("Image[85000] = %f\n", GrayImageData[85000]);
        for (size_t row = 0; row < filterSize; ++row)
        {
            printf("Row=%ld of Gaussian filter = ", row);
            for (size_t col = 0; col < filterSize; ++col)
            {
                printf("%f ", filter[col + filterSize * row]);
            }
            printf("\n");
        }
        printf("Otsu's Threshold = %f\n", thresh);
        printf("\n");
    #endif


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
