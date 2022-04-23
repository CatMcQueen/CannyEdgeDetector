#include "Otsus_Method.h"
#include <cmath>

#define OUTPUT_VAL 200
#define NUM_BINS 256

void Histogram_Sequential(float *image, unsigned int *hist, int width, int height)
{
	int pos = 0;

	// Loop through every pixel
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (image[row*width + col] > 1)
			{
				pos = 255;
			}
			else
			{
				pos = int(image[row*width + col] * 255);
			}

			// Update histogram
			hist[pos]++;
		}

	}

}

double Otsu_Sequential(unsigned int* histogram, int width, int height)
{

	float bin_mids[256];
	float histogram_bin_mids[256];
	float weight1[256];
	float weight2[256];
	float cumsum_mean1[256];
	float cumsum_mean2[256];
	float mean1[256];
	float mean2[256];
	float inter_class_variance[255];
	float max_variance = 0;

	int thresh = 0;

	float bin_length = 255.0f/256.0f;
	float half_bin_length = 255.0f/512.0f;

	// Calculate bin mids
	for(int i = 0; i < 256; i++)
	{
		bin_mids[i] = half_bin_length + bin_length * i;
		histogram_bin_mids[i] = histogram[i] * (half_bin_length + bin_length * i);
	}

	weight1[0] = histogram[0];
	weight2[0] = width * height;

	// Calculate class probabilities
	for(int i = 1; i < 256; i++)
	{
		weight1[i] = histogram[i] + weight1[i-1];
		weight2[i] = weight2[i-1] - histogram[i-1];
	}

	cumsum_mean1[0] = histogram_bin_mids[0];
	cumsum_mean2[0] = histogram_bin_mids[255];

	// Calculate class means
	for(int i = 1; i < 256; i++)
	{
		cumsum_mean1[i] = cumsum_mean1[i-1] + histogram_bin_mids[i];
		cumsum_mean2[i] = cumsum_mean2[i-1] + histogram_bin_mids[256 - i - 1];
		mean1[i] = cumsum_mean1[i] / weight1[i];
		mean2[256 - i - 1] = cumsum_mean2[i] / weight2[256 - i - 1];
	}

	// Calculate Inter_class_variance
	for(int i = 0; i < 255; i++)
	{
		inter_class_variance[i] = (weight1[i] * weight2[i] * (mean1[i] - mean2[i+1])) * (mean1[i] - mean2[i+1]);	
	}

	// Maximize interclass variance
	for(int i = 0;i < 255; i++){
		if(max_variance < inter_class_variance[i])
		{
			max_variance = inter_class_variance[i];
			thresh = i;
		}
	}

	// Return normalized threshold
	//return bin_mids[thresh]; //This is the actual Otsu's threshold
	//return cumsum_mean1[OUTPUT_VAL]; //This is a test value
	return thresh; // This is also a test value and equivalent to key[0]

}

__global__ void NaiveHistogram(float* image, unsigned int* histogram, int width, int height)
{
	// insert your code here
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = blockDim.x * gridDim.x;

	while(tid < width * height)
	{
		int position = int(image[tid]*255);

		if (position >= 0 && position < 256)
		{
			atomicAdd(&(histogram[position]),1);
		}

		tid += stride;

	}
}

__global__ void OptimizedHistogram(float* image, unsigned int* histogram, int width, int height)
{
	__shared__ unsigned int histogram_private[256];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = blockDim.x * gridDim.x;

	for(int bin = threadIdx.x; bin < 256; bin += blockDim.x)
	{
		histogram_private[bin] = 0;
	}

	__syncthreads();

	while(tid < width * height)
	{
		int position = int(image[tid]*255);

		if (position >= 0 && position < 256)
		{
			atomicAdd(&(histogram_private[position]),1);
		}

		tid += stride;

	}

	__syncthreads();

	for(int bin = threadIdx.x; bin < 256; bin += blockDim.x)
	{
		atomicAdd(&(histogram[bin]), histogram_private[bin]);
	}

}

__global__ void NaiveOtsu(unsigned int *histogram, float* thresh, int width, int height)
{
	__shared__ float weight1[256];
	__shared__ float weight2[256];

	__shared__ float bin_mids[256];
	__shared__ float histogram_bin_mids[256];

	__shared__ float mean1[256];
	__shared__ float mean2[257];

	__shared__ float inter_class_variance[256];
	__shared__ int key[256];

	float bin_length = 0.99609375;
	float half_bin_length = 0.498046875;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < 256)
	{
		bin_mids[tid] = half_bin_length + bin_length * tid;
		histogram_bin_mids[tid] = histogram[tid] * (half_bin_length + bin_length * tid);

		__syncthreads();

		float w1 = histogram[0];
		float w2 = width * height;

		float cs_mean1 = histogram_bin_mids[0];
		float cs_mean2 = histogram_bin_mids[255];

		// Calculate class probabilities and means
		for(int i = 1; i < tid + 1; i++)
		{
			w1 += histogram[i];
			w2 -= histogram[i-1];
			cs_mean1 += histogram_bin_mids[i];
			cs_mean2 += histogram_bin_mids[256-i-1];
		}

		weight1[tid] = w1;
		weight2[tid] = w2;

		__syncthreads();

		mean1[tid] = cs_mean1 / weight1[tid];
		mean2[256 - tid - 1] = cs_mean2 / weight2[256 - tid - 1];

		if (tid == 0)
		{
			mean1[0] = 0;
		}

		if (tid == 255)
		{
			mean2[255] = 0;
		}
	
		key[tid] = tid;

		__syncthreads();

		inter_class_variance[tid] = (weight1[tid] * weight2[tid] * (mean1[tid] - mean2[tid+1])) * (mean1[tid] - mean2[tid+1]);

		for (int stride = 1; stride < 256; stride *= 2)
		{
			if(tid % (2*stride) == 0)
			{
				if(inter_class_variance[tid] < inter_class_variance[tid+stride])
				{
					inter_class_variance[tid] = inter_class_variance[tid+stride];
					key[tid] = key[tid+stride];
				}
			}
			__syncthreads();
		}
	
		__syncthreads();

		if(tid == 0)
		{
			thresh[0] = bin_mids[key[0]];
			//thresh[0] = key[0]; // Test value
		}
	}
}


__global__ void OptimizedOtsu(unsigned int *histogram, float* thresh, int width, int height)
{
	__shared__ unsigned int weight1[256];
	__shared__ unsigned int weight2[256];

	__shared__ float mean1[256];
	__shared__ float mean2[257];

	__shared__ float inter_class_variance[256];
	__shared__ int key[256];

	__shared__ unsigned int hist_private[256];

	double bin_length = 255.0f/256.0f;
	double half_bin_length = 255.0f/512.0f;

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < 256)
	{
		hist_private[tid] = histogram[tid];

		__syncthreads();

		weight1[tid] = hist_private[tid];

		mean1[tid] = hist_private[tid] * (half_bin_length + bin_length * tid);
		mean2[255-tid] = hist_private[tid] * (half_bin_length + bin_length * tid);

		//__syncwarp();
		__syncthreads();

		float cs_mean1 = mean1[0];
		float cs_mean2 = mean1[255];

		// Calculate class probabilities and means
		for(int i = 1; i < tid + 1; i++)
		{
			cs_mean1 += mean1[i];
			cs_mean2 += mean1[256-i-1];
		}

		for(int stride = 1; stride <= tid; stride = stride * 2)
		{
			unsigned int w1 = weight1[tid - stride];
			float m1 = mean1[tid - stride];
			float m2 = mean2[tid - stride];
			weight1[tid] += w1;
			mean1[tid] += m1;
			mean2[tid] += m2;
		}

		__syncwarp();
		__syncthreads();

		//weight2[tid] = width * height - weight1[tid] + hist_private[tid];

		//__syncwarp();
		//__syncthreads();
		//
		////float cs_mean1 = mean1[tid];
		////float cs_mean2 = mean2[tid];

		//__syncwarp();
		//__syncthreads();

		//mean1[tid] = cs_mean1 / weight1[tid];
		//mean2[tid] = cs_mean2 / weight2[255-tid];		

		//if (tid == 0)
		//{
		//	mean1[0] = 0;
		//}

		//if (tid == 255)
		//{
		//	mean2[255] = 0;
		//}
	
		//// Make an ordered vector 0-255
		//key[tid] = tid;

		//__syncwarp();
		//__syncthreads();

		//inter_class_variance[tid] = (weight1[tid] * weight2[tid] * (mean1[tid] - mean2[tid+1])) * (mean1[tid] - mean2[tid+1]) * 0.0000001f;

		//__syncwarp();
		//__syncthreads();

		//for (int stride = 1; stride < 256; stride *= 2)
		//{
		//	if(tid % (2*stride) == 0)
		//	{
		//		if(inter_class_variance[tid] < inter_class_variance[tid+stride])
		//		{
		//			inter_class_variance[tid] = inter_class_variance[tid+stride];
		//			key[tid] = key[tid+stride];
		//		}
		//	}
		//	__syncthreads();
		//}

		//__syncwarp();
		//__syncthreads();

		////key[0] should not be 0 if working properly

		if(tid == OUTPUT_VAL)
		{
			//thresh[0] = half_bin_length + bin_length * key[0]; //This is the actual Otsu's threshold
			//thresh[0] = cs_mean1; //This is a test value

			//thresh[0] = key[0];
			//thresh[0] = hist_private[0];
			thresh[0] = mean1[0];
		}

	}	

}
