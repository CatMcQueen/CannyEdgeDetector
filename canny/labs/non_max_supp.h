#ifndef __NON_MAX_SUPP_H__
#define __NON_MAX_SUPP_H__

void nms(float *inImg, float *nmsImg, float *gradImg, int height, int width);

#ifdef __CUDACC__
__global__
void nms_global(float *inImg, float *nmsImg, float *gradImg, int height, int width);

__global__
void nms_opt(float *inImg, float *nmsImg, float *gradImg, int height, int width);
#endif // __CUDACC__

#endif // __NON_MAX_SUPP_H__
