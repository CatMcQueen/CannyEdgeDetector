// To calculate the Grayscale image = ColorToGrayscale
// For the gaussian blur = Conv2D
// and the sobel filter which gives the 
//  gradient descent = GradientSobel
#include "filters.h"

void populate_blur_filter(double *outFilter, size_t filterEdgeLen)
{
    //double scaleVal = 1;
    //double stDev = (double)FILTERSIZE/3;

    double stDevSq = 0.6;
    double pi = M_PI;
	 double scaleFac = (1 / (2*pi*stDevSq));

    for (int i = 0; i < filterEdgeLen; ++i) {
        for (int j = 0; j < filterEdgeLen; ++j) {

			// pow() is slow so just multiply out
            double xComp = (i + 1 - (filterEdgeLen+1)/2) * (i + 1 - (filterEdgeLen+1)/2);
            double yComp = (j + 1 - (filterEdgeLen+1)/2) * (j + 1 - (filterEdgeLen+1)/2);

            //calculate the value at each index of the Kernel
            double filterVal = exp(-(xComp + yComp) / (2 * stDevSq));
            filterVal = scaleFac * filterVal;

            //populate Kernel
            outFilter[i + j*filterEdgeLen] = filterVal;

	/*
            if (i==0 && j==0)
            {
                scaleVal = outFilter[0][0];
            }

            //normalize Kernel
            outFilter[i][j] = outFilter[i][j] / scaleVal;
	*/			

        }
    }
}


void ColorToGrayscaleSerial(float *input, float *output,
                    unsigned int y, unsigned int x) {
  for (unsigned int ii = 0; ii < y; ii++) {
    for (unsigned int jj = 0; jj < x; jj++) {
      unsigned int idx = ii * x + jj;
      float r          = input[3 * idx];     // red value for pixel
      float g          = input[3 * idx + 1]; // green value for pixel
      float b          = input[3 * idx + 2];
      output[idx] = (float)(0.21f * r + 0.71f * g + 0.07f * b);
    }
  }
}

// convert the image to grayscale
__global__ void ColorToGrayscale(float *inImg, float *outImg, int width, int height) {
   int idx, grayidx;
   int col = blockDim.x * blockIdx.x + threadIdx.x;
   int row  = blockDim.y * blockIdx.y + threadIdx.y;
   int numchannel = 3;

   // x = col and y = row
   if (col >= 0 && col < width && row >=0 && row < height) {
      // each spot is 3 big (rgb) so get the number of spots
      grayidx = row * width + col;
      idx     = grayidx * numchannel; // and multiply by three
      // to calculate the beginning of the 3 for that pixel
      float r = inImg[idx];           //red
      float g = inImg[idx + 1];       //green
      float b = inImg[idx + 2];       //blue
      outImg[grayidx]  = (0.21*r + 0.71*g + 0.07*b);
   }
}



// the gaussian blur is just a conv2d with a filter
__global__ void Conv2D(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int halfFilter = (int)filterSize/2;

   // boundary check if it's in the image
   if(row >= 0 && row < height && col >= 0 && col < width) {
     float pixelvalue = 0;
     int start_col = col - halfFilter;
     int start_row = row - halfFilter;

     // now do the filtering
     for (int j = 0; j < filterSize; ++j) {
       for (int k = 0; k < filterSize; ++k) {
         int cur_row = start_row + j;
         int cur_col = start_col + k;

         // only count the ones that are inside the boundaries
         if (cur_row >=0 && cur_row < height && cur_col >= 0 && cur_col < width) {
           pixelvalue += inImg[cur_row*width + cur_col] * filter[j + k*filterSize];
         }

       }
     }
     __syncthreads();
     outImg[row*width + col] = pixelvalue;      
   }

}

__global__ void GradientSobel(float *inImg, float *sobelImg, float *gradientImg, int height, int width, size_t filterSize) {
    //int filterSize = (int)FILTERSIZE;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // To detect horizontal lines, G_x. 
    const int fmat_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    // To detect vertical lines, G_y 
    const int fmat_y[3][3]  = {
        {-1, -2, -1},
        {0,   0,  0},
        {1,   2,  1}
    };

    // now do the filtering
    // halfFitler is how many are on each side
    int halfFilter = (int)filterSize/2;
    double sumx = 0;
    double sumy = 0;
    //// DO THE SOBEL FILTERING ///////////

    // boundary check if it's in the image
    if(row >= 0 && row < height && col >= 0 && col < width) {
        int start_col = col - halfFilter;
        int start_row = row - halfFilter;

        // now do the filtering
        for (int j = 0; j < filterSize; ++j) {
            for (int k = 0; k < filterSize; ++k) {
                int cur_row = start_row + j;
                int cur_col = start_col + k;

                // only count the ones that are inside the boundaries
                if (cur_row >=0 && cur_row < height && cur_col >= 0 && cur_col < width ) {
                    sumy += inImg[cur_row*width + cur_col] * fmat_y[j][k];
                    sumx += inImg[cur_row*width + cur_col] * fmat_x[j][k];
                }
            }
        }

        // now calculate the sobel output and gradients
        sobelImg[row*width + col] = sqrt(sumx * sumx + sumy*sumy); // output of the sobel filter
        gradientImg[row*width + col] = atan(sumx/sumy) * 180/M_PI; // the gradient calculateion
    }
 
}


void Conv2DSerial(float *inImg, float *outImg, double *filter, int width, int height, size_t filterSize) {

    // find center position of kernel (half of kernel size)
    int filterHalf = filterSize / 2;
    
    // iterate over rows and coluns of the image
    for(int row=0; row < height; ++row)              // rows
    {
        for(int col=0; col < width; ++col)          // columns
        {
            int start_col = col - filterHalf;
            int start_row = row - filterHalf;
            float pixelvalue = 0; 

            // then for each pixel iterate through the filter
            for(int j=0; j < filterSize; ++j)     // filter rows
            {
                for(int k=0; k < filterSize; ++k) // kernel columns
                {
                    int cur_row = start_row + j;
                    int cur_col = start_col + k;
                    if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width) {
                        pixelvalue += inImg[cur_row*width + cur_col] * filter[j + k*filterSize];
                    }
                }
            }
            outImg[row*width+col] = pixelvalue;
        }
    }
}


void GradientSobelSerial(float *inImg, float *mag, float *phase, int height, int width, size_t filterSize)
{

	//int filterSize = (int)FILTERSIZE;
	int halfFilter = (int)(filterSize)/2;
	
	// To detect horizontal lines, G_x. 
	const int fmat_x[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	// To detect vertical lines, G_y 
	const int fmat_y[3][3]  = {
		{-1, -2, -1},
		{0,   0,  0},
		{1,   2,  1}
	};

	// iterate over rows and columns of the image
	for(int row=0; row < height; ++row)              // rows
	{
		for(int col=0; col < width; ++col)          // columns
		{

			double sumx = 0;
			double sumy = 0;

			int start_col = col - halfFilter;
			int start_row = row - halfFilter;

			// now do the filtering
			for (int j = 0; j < filterSize; ++j)
			{
				for (int k = 0; k < filterSize; ++k)
				{

					int cur_row = start_row + j;
					int cur_col = start_col + k;

					// only count the ones that are inside the boundaries
					if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width)
					{
						sumy += inImg[cur_row*width + cur_col] * fmat_y[j][k];
						sumx += inImg[cur_row*width + cur_col] * fmat_x[j][k];
					}

				}
			}

			mag[row*width + col] = sqrt(sumx * sumx + sumy * sumy); // output of the sobel filt at this index
			phase[row*width+col] = atan(sumx/sumy) * 180/M_PI; // gradient at pixel

		}
	}
}
