#include "non_max_supp.h"
//#include <stdio.h>

__host__ __device__
float maxSupp(float center, float p1, float p2, float p3=-1.0, float p4=-1.0)
{
  if (center >= p1 && center >= p2 && center >= p3 && center >= p4)
  {
    return center;
  }
  else
  {
    return 0.0;
  }
}

__host__ __device__
float getPoint(float *img, int cIdx, int rIdx, int height, int width)
{
  if (!img || rIdx < 0 || rIdx >= width || cIdx < 0 || cIdx >= height)
  {
    return 0.0;
  }
  return *(img + cIdx + rIdx*width);
}

void nms(float *inImg, float *nmsImg, float *gradImg, int height, int width)
{
  //FILE *quantFile = fopen("quantNms.txt", "w");

  for(int j = 0; j < height; ++j)
  {
    for(int i = 0; i < width; ++i)
    {
      float angle = *(gradImg + j*width + i);
      float p1 = -1.0;//, p3 = -1.0;
      float p2 = -1.0;//, p4 = -1.0;
      unsigned int fAngle = 0;
      if (angle > 180)
      {
        angle = angle - 180;
      }

      //if ((angle > 0 && angle <= 22.5) || (angle > 157.5 && angle <= 180))
      //  fAngle = 0;
      //else if (angle > 22.5 && angle <= 67.5)
      //  fAngle = 45;
      //else if (angle > 67.5 && angle <= 112.5)
      //  fAngle = 90;
      //else if (angle > 112.5 && angle <= 157.5)
      //  fAngle = 135;

      if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5) || (angle < -157.5))
        fAngle = 0;
      else if ((angle > 112.5 && angle <= 157.5) || (angle < -22.5 && angle >= -67.5))
        fAngle = 135;
      else if ((angle > 67.5 && angle <= 112.5) || (angle < -67.5 && angle >= -112.5))
        fAngle = 90;
      else if ((angle > 22.5 && angle <= 67.5) || (angle < -112.5 && angle >= -157.5))
        fAngle = 45;
      
      //fprintf(quantFile, "%d,", fAngle);
      switch( fAngle ) 
      {
        case 0:
          p1 = getPoint(inImg, i, j+1, height, width);
          p2 = getPoint(inImg, i, j-1, height, width);
          //p3 = getPoint(inImg, i, j+2, height, width);
          //p4 = getPoint(inImg, i, j-2, height, width);
          break;
        case 45:
          p1 = getPoint(inImg, i-1, j-1, height, width);
          p2 = getPoint(inImg, i+1, j+1, height, width);
          //p3 = getPoint(inImg, i-2, j-2, height, width);
          //p4 = getPoint(inImg, i+2, j+2, height, width);
          break;
        case 90:
          p1 = getPoint(inImg, i+1, j, height, width);
          p2 = getPoint(inImg, i-1, j, height, width);
          //p3 = getPoint(inImg, i+2, j, height, width);
          //p4 = getPoint(inImg, i-2, j, height, width);
          break;
        case 135:
          p1 = getPoint(inImg, i+1, j-1, height, width);
          p2 = getPoint(inImg, i-1, j+1, height, width);
          //p3 = getPoint(inImg, i+2, j-2, height, width);
          //p4 = getPoint(inImg, i-2, j+2, height, width);
          break;
        default:
          break;
      }

      float center = getPoint(inImg, i, j, height, width);
      //*(nmsImg + i + j*width) = maxSupp(center, p1, p2, p3, p4);
      *(nmsImg + i + j*width) = maxSupp(center, p1, p2);
    }
    //fprintf(quantFile, "\n");
  }
  //fclose(quantFile);
}

__global__ 
void nms_global(float *inImg, float *nmsImg, float *gradImg, int height, int width)
{
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  size_t row  = blockDim.y * blockIdx.y + threadIdx.y;

  float p1 = -1.0;//, p3 = -1.0;
  float p2 = -1.0;//, p4 = -1.0;
  unsigned int fAngle = 0;

  //if (col >= 0 && col < width && row >= 0 && row < height)
  if (col < width && row < height) // Since size_t is unsigned, it can't fall below 0
  {
    float angle = *(gradImg + row*width + col);

    if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5) || (angle < -157.5))
      fAngle = 0;
    else if ((angle > 112.5 && angle <= 157.5) || (angle < -22.5 && angle >= -67.5))
      fAngle = 135;
    else if ((angle > 67.5 && angle <= 112.5) || (angle < -67.5 && angle >= -112.5))
      fAngle = 90;
    else if ((angle > 22.5 && angle <= 67.5) || (angle < -112.5 && angle >= -157.5))
      fAngle = 45;

    switch (fAngle)
    {
      case 0:
        p1 = getPoint(inImg, col, row+1, height, width);
        p2 = getPoint(inImg, col, row-1, height, width);
        //p3 = getPoint(inImg, col, row+2, height, width);
        //p4 = getPoint(inImg, col, row-2, height, width);
        break;
      case 45:
        p1 = getPoint(inImg, col-1, row-1, height, width);
        p2 = getPoint(inImg, col+1, row+1, height, width);
        //p3 = getPoint(inImg, col-2, row-2, height, width);
        //p4 = getPoint(inImg, col+2, row+2, height, width);
        break;
      case 90:
        p1 = getPoint(inImg, col+1, row, height, width);
        p2 = getPoint(inImg, col-1, row, height, width);
        //p3 = getPoint(inImg, col+2, row, height, width);
        //p4 = getPoint(inImg, col-2, row, height, width);
        break;
      case 135:
        p1 = getPoint(inImg, col+1, row-1, height, width);
        p2 = getPoint(inImg, col-1, row+1, height, width);
        //p3 = getPoint(inImg, col+2, row-2, height, width);
        //p4 = getPoint(inImg, col-2, row+2, height, width);
        break;
      default:
        break;
    }

    float center = getPoint(inImg, col, row, height, width);
    //*(nmsImg + i + j*width) = maxSupp(center, p1, p2, p3, p4);
    *(nmsImg + col + row*width) = maxSupp(center, p1, p2);
  }
}

__global__
void nms_opt(float *inImg, float *nmsImg, float *gradImg, int height, int width)
{
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  size_t row  = blockDim.y * blockIdx.y + threadIdx.y;

  float p1 = -1.0;//, p3 = -1.0;
  float p2 = -1.0;//, p4 = -1.0;

  if (col < width && row < height) // Since size_t is unsigned, it can't fall below 0
  {
    float angle = *(gradImg + row*width + col);

    if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5) || (angle < -157.5))
    {
        p1 = getPoint(inImg, col, row+1, height, width);
        p2 = getPoint(inImg, col, row-1, height, width);
        //p3 = getPoint(inImg, col, row+2, height, width);
        //p4 = getPoint(inImg, col, row-2, height, width);
    }
    else if ((angle > 112.5 && angle <= 157.5) || (angle < -22.5 && angle >= -67.5))
    {
        p1 = getPoint(inImg, col+1, row-1, height, width);
        p2 = getPoint(inImg, col-1, row+1, height, width);
        //p3 = getPoint(inImg, col+2, row-2, height, width);
        //p4 = getPoint(inImg, col-2, row+2, height, width);
    }
    else if ((angle > 67.5 && angle <= 112.5) || (angle < -67.5 && angle >= -112.5))
    {
        p1 = getPoint(inImg, col+1, row, height, width);
        p2 = getPoint(inImg, col-1, row, height, width);
        //p3 = getPoint(inImg, col+2, row, height, width);
        //p4 = getPoint(inImg, col-2, row, height, width);
    }
    else if ((angle > 22.5 && angle <= 67.5) || (angle < -112.5 && angle >= -157.5))
    {
        p1 = getPoint(inImg, col-1, row-1, height, width);
        p2 = getPoint(inImg, col+1, row+1, height, width);
        //p3 = getPoint(inImg, col-2, row-2, height, width);
        //p4 = getPoint(inImg, col+2, row+2, height, width);
    }

    float center = getPoint(inImg, col, row, height, width);
    //*(nmsImg + i + j*width) = maxSupp(center, p1, p2, p3, p4);
    *(nmsImg + col + row*width) = maxSupp(center, p1, p2);
  }
}

