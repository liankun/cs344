// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset
  size_t index_x = threadIdx.x+blockIdx.x*blockDim.x;
  size_t stride_x = gridDim.x*blockDim.x;
  size_t index_y = threadIdx.y+blockIdx.y*blockDim.y;
  size_t stride_y = gridDim.y*blockDim.y;
  for(size_t i=index_x;i<numRows;i+=stride_x){
    for(size_t j=index_y;j<numCols;j+=stride_y){
       uchar4 rgba = rgbaImage[i*numCols+j];
       float channelSum = 0.299f * rgba.x+0.587f * rgba.y + 0.114f * rgba.z;
       greyImage[i * numCols + j] = channelSum;  
//       if(i*numCols +j==905) printf("pos %d val %f\n",i*numCols +j,channelSum);
//       printf("GPU id %d %d\n",index_x,index_y);
    }
  } 
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  printf("perform my rgba convert function !\n");
  //get GPU information
  int deviceId;
  int numberOfSMs;
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs,cudaDevAttrMultiProcessorCount,deviceId);  
  
  printf("device id %d number of SMs %d\n",deviceId,numberOfSMs);

  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  int numberOfBlocks = numberOfSMs*1;
  int threadPerblock = 16;
  const dim3 blockSize(threadPerblock, threadPerblock, 1);  //TODO
  const dim3 gridSize( numberOfBlocks, numberOfBlocks, 1);  //TODO
  rgba_to_greyscale <<<gridSize, blockSize>>> (d_rgbaImage, d_greyImage, numRows, numCols);
  
  
  cudaDeviceSynchronize();
//  if(asynError!=cudaSuccess) printf("asynError %s\n",cudaGetErrorString(asynError));
//  if(asynError!=cudaSuccess)

  checkCudaErrors(cudaGetLastError());

//  cudaError_t synError; 
//  synError = cudaGetLastError();
//  if(synError!=cudaSuccess) printf("asynError %s\n",cudaGetErrorString(synError));

//  checkCudaErrors(synError);

}
