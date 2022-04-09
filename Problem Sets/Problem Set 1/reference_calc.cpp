// for uchar4 struct
#include <cuda_runtime.h>
#include <stdio.h>

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols)
{
  for (size_t r = 0; r < numRows; ++r) {
    for (size_t c = 0; c < numCols; ++c) {
      uchar4 rgba = rgbaImage[r * numCols + c];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[r * numCols + c] = channelSum;
//      if(r * numCols + c==905) printf("ref pos %d val %f\n",r * numCols + c,channelSum);
    }
  }
}

