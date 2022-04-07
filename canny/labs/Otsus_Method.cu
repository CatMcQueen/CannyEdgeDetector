#include "Otsus_Method.h"
#include <cmath>

void Histogram_Sequential(float *image, unsigned int *hist, int width, int height)
{
	// Row pointer
	float* matrix = image;

	// Loop through every pixel
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// Calculate position
			int pos = floor(matrix[j]*255 + 0.5);
			
			// Update histogram
			hist[pos]++;
		}

		// Update row pointer
		matrix += height;

	}

}

double Otsu_Sequential(unsigned int* histogram)
{

	// Calculate the bin_edges
	double bin_edges[256];
	bin_edges[0] = 0.0;
	double increment = 0.99609375;
	for (int i = 1; i < 256; i++)
		bin_edges[i] = bin_edges[i - 1] + increment;

	// Calculate bin_mids
	double bin_mids[256];
	for (int i = 0; i < 256; i++)
		bin_mids[i] = (bin_edges[i] + bin_edges[i + 1]) / 2;

	// Iterate over all thresholds (indices) and get the probabilities weight1, weight2
	double weight1[256];
	weight1[0] = histogram[0];
	for (int i = 1; i < 256; i++)
		weight1[i] = histogram[i] + weight1[i - 1];

	int total_sum = 0;
	for (int i = 0; i < 256; i++)
		total_sum = total_sum + histogram[i];
	double weight2[256];
	weight2[0] = total_sum;
	for (int i = 1; i < 256; i++)
		weight2[i] = weight2[i - 1] - histogram[i - 1];

	// Calculate the class means: mean1 and mean2
	double histogram_bin_mids[256];
	for (int i = 0; i < 256; i++)
		histogram_bin_mids[i] = histogram[i] * bin_mids[i];

	double cumsum_mean1[256];
	cumsum_mean1[0] = histogram_bin_mids[0];
	for (int i = 1; i < 256; i++)
		cumsum_mean1[i] = cumsum_mean1[i - 1] + histogram_bin_mids[i];

	double cumsum_mean2[256];
	cumsum_mean2[0] = histogram_bin_mids[255];
	for (int i = 1, j = 254; i < 256 && j >= 0; i++, j--)
		cumsum_mean2[i] = cumsum_mean2[i - 1] + histogram_bin_mids[j];

	double mean1[256];
	for (int i = 0; i < 256; i++)
		mean1[i] = cumsum_mean1[i] / weight1[i];

	double mean2[256];
	for (int i = 0, j = 255; i < 256 && j >= 0; i++, j--)
		mean2[j] = cumsum_mean2[i] / weight2[j];

	// Calculate Inter_class_variance
	double Inter_class_variance[255];
	double dnum = 10000000000;
	for (int i = 0; i < 255; i++)
		Inter_class_variance[i] = ((weight1[i] * weight2[i] * (mean1[i] - mean2[i + 1])) / dnum) * (mean1[i] - mean2[i + 1]);

	// Maximize interclass variance
	double maxi = 0;
	int getmax = 0;
	for (int i = 0; i < 255; i++) {
		if (maxi < Inter_class_variance[i]) {
			maxi = Inter_class_variance[i];
			getmax = i;
		}
	}

	return bin_mids[getmax];

}

__global__ void NaiveHistogram(unsigned char *image, int width, int height, int *hist)
{
    // Calculate threadID in x and y directions
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate linear threadID
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Calculate number of threads in x and y directions
    int num_threads_x = blockDim.x * gridDim.x;
    int num_threads_y = blockDim.y * gridDim.y;

    // Calculate linear blockID
    int bid = blockIdx.x + blockIdx.y * gridDim.x;

    // Loop through all pixels
    for (int col = tid_x; col < width; col += num_threads_x)
    {
        for (int row = tid_y; row < height; row += num_threads_y)
        {
            unsigned char pos = image[tid];
            atomicAdd(&hist[pos], 1);
        }
    }

}