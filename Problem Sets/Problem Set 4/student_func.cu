//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <string>
#include <thrust/host_vector.h>
#include <vector>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__
void checkBit(unsigned int* const d_inputVals,
	      unsigned int* d_bitValueOne,
	      unsigned int* d_bitValueZero,
	      const unsigned int bitPos,
	      const int nsize
	      ){
  int index = threadIdx.x+blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;

  for(int i=index;i<nsize;i+=stride){
     unsigned int mask = (2 - 1) << bitPos;
     d_bitValueOne[i] = (d_inputVals[i] & mask)>>bitPos;
     d_bitValueZero[i] = 1 - d_bitValueOne[i];
     if(d_bitValueOne[i]>1) printf("index %d bit value: %d\n",i,d_bitValueOne[i]);
  }

}

__global__
void gpuCountReduce(const unsigned int* d_bitValue,
		    unsigned int* count,
		    const size_t nsize,
		    bool sum_one = true){
  int index = threadIdx.x+blockIdx.x*blockDim.x;
  int tid = threadIdx.x;
  extern __shared__ unsigned int sdata[];
  //count zero first
  if(index<nsize){
    if(sum_one) sdata[tid] = d_bitValue[index];
    else sdata[tid] = 1-d_bitValue[index];
  }
  else sdata[tid] = 0;
  
  __syncthreads();
  
  int s = blockDim.x/2;
  for(int i=tid;i<s;s>>=1){
    sdata[tid]+=sdata[tid+s];
    __syncthreads(); 
  }

  if(tid==0){
    count[blockIdx.x] = sdata[0];
//    printf("block id %d count %d \n",blockIdx.x,sdata[0]);
  }
  
}

void countReduce(const unsigned int* d_bitValue,
		 unsigned int* count,
		 const size_t nsize
		){
   //*count is a number
   int threadPerBlock =1024;
   int t_nsize = nsize;
   int numberOfBlock = t_nsize/threadPerBlock;
   if(t_nsize > threadPerBlock*numberOfBlock) numberOfBlock+=1;
   unsigned int* d_m_count=count;
   if(numberOfBlock>1){ 
     checkCudaErrors(cudaMalloc(&d_m_count,sizeof(unsigned int)*numberOfBlock));
   }

 //  printf("Count reduce: Nblocks %d Nthreads %d\n",numberOfBlock,threadPerBlock);
   gpuCountReduce<<<numberOfBlock,threadPerBlock,sizeof(unsigned int)*threadPerBlock>>>(d_bitValue,
		                                                                        d_m_count,
											t_nsize);
   checkCudaErrors(cudaDeviceSynchronize());
   checkCudaErrors(cudaGetLastError());
   if(numberOfBlock==1) return;

   t_nsize = numberOfBlock;
   numberOfBlock=t_nsize/threadPerBlock;
   if(t_nsize > threadPerBlock*numberOfBlock) numberOfBlock+=1;
   
   unsigned int* d_in = d_m_count;
   d_m_count = count;
   if(numberOfBlock>1){
     checkCudaErrors(cudaMalloc(&d_m_count,sizeof(unsigned int)*numberOfBlock));
   }
   gpuCountReduce<<<numberOfBlock,threadPerBlock,sizeof(unsigned int)*threadPerBlock>>>(d_in,
		                                                                        d_m_count,
											t_nsize);
   if(numberOfBlock==1){
     checkCudaErrors(cudaFree(d_in));
     return;
   }


   while(numberOfBlock>1){
     t_nsize = numberOfBlock;
     numberOfBlock=t_nsize/threadPerBlock;
     if(t_nsize > threadPerBlock*numberOfBlock) numberOfBlock+=1;
     std::swap(d_in,d_m_count);
     gpuCountReduce<<<numberOfBlock,threadPerBlock,sizeof(unsigned int)*threadPerBlock>>>(d_in,
		                                                                          d_m_count,
											  t_nsize);
   }


   //count[0] = d_m_count[0];  
   checkCudaErrors(cudaMemcpy(count,d_m_count,sizeof(unsigned int)*1,cudaMemcpyDeviceToDevice));
//   printf("GPU finish succesfully\n");
}

__global__
void gpuScanInit(const unsigned int* d_bitValue,
		 unsigned int* d_media,
		 unsigned int* d_ouput,
		 const size_t nsize){
  //init the value
  //exclusive
  int index = threadIdx.x+blockIdx.x*blockDim.x;
  if(index<nsize){
    if(index>0){
      d_media[index] = d_bitValue[index-1];
    }
    else d_media[index] = 0;
    d_ouput[index] = 0;
  }
}

__global__
void gpuScanAcc(unsigned int* d_media,
		unsigned int* d_output,
		const size_t numberOfSum,
		const size_t nsize){
  //accumulate prefix  
  size_t index = threadIdx.x+blockIdx.x*blockDim.x;
  if(index<nsize){
    d_output[index] = d_media[index];
    if(numberOfSum<index) d_output[index]+=d_media[index-numberOfSum];
  }
}

__global__
void gpuScanAdd(const unsigned int* d_bitValue,
		unsigned int* d_pos,
		const unsigned int* d_output,
		const unsigned int shift,
		const size_t nsize
		){
  //add shift to get new position
  unsigned int index = threadIdx.x+blockIdx.x*blockDim.x;
  if(index<nsize){
    if(d_bitValue[index]==1){
      d_pos[index]=shift+d_output[index];
      //if(index<20) printf("old pos %d new pos %d\n",index,d_pos[index]);
    }
  }
}

void scan(unsigned int* d_pos,
	  const unsigned int* d_bitValue,
	  const unsigned int shift,
	  const size_t nsize){
  //sum one indicate the scan bit 1 or 0
  unsigned int* d_media;
  unsigned int* d_output;
  checkCudaErrors(cudaMalloc(&d_media,sizeof(unsigned int)*nsize));
  checkCudaErrors(cudaMalloc(&d_output,sizeof(unsigned int)*nsize)); 
  unsigned int threadPerBlock = 1024;
  unsigned int numberOfBlock = nsize/threadPerBlock;
  if(numberOfBlock*threadPerBlock<nsize) numberOfBlock+=1;
  gpuScanInit<<<numberOfBlock,threadPerBlock>>>(d_bitValue,
		                                d_media,
						d_output,
						nsize);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  unsigned int numberOfSum=1;
  unsigned int* d_i = d_media;
  unsigned int* d_o = d_output;

  while(numberOfSum<nsize-1){
    gpuScanAcc<<<numberOfBlock,threadPerBlock>>>(d_i,d_o,numberOfSum,nsize);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    numberOfSum+=numberOfSum;
    std::swap(d_i,d_o);
  }
  //get the index after the add the shift
//  std::swap(d_i,d_o);
  gpuScanAdd<<<numberOfBlock,threadPerBlock>>>(d_bitValue,d_pos,d_i,shift,nsize);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_media));
  checkCudaErrors(cudaFree(d_output));
}

__global__
void gpuMapToPos(const unsigned int* d_pos,
		 unsigned int* d_inputVals,
		 unsigned int* d_inputPos,
		 unsigned int* d_outputVals,
		 unsigned int* d_outputPos,
		 const size_t numElems){
  //don't use the size_t for indexing
  unsigned int index = threadIdx.x+blockIdx.x*blockDim.x;
  if(index<numElems){
    d_outputVals[d_pos[index]] = d_inputVals[index];
    d_outputPos[d_pos[index]] = d_inputPos[index];

 //   if(index<10){
 //     printf("map old index %d new index %d \n",index,d_pos[index]); 
 //   }
  }
}

__global__
void gpuCopy(unsigned int* d_inputVals,
	     unsigned int* d_inputPos,
	     unsigned int* d_outputVals,
	     unsigned int* d_outputPos,
	     const size_t numElems){
  unsigned int index = threadIdx.x+blockIdx.x*blockDim.x;
  if(index<numElems){
    d_outputVals[index] = d_inputVals[index];
    d_outputPos[index] = d_inputPos[index];
  }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  unsigned int* d_srcVals = d_inputVals;
  unsigned int* d_srcPos = d_inputPos;
  unsigned int* d_dstVals = d_outputVals;
  unsigned int* d_dstPos = d_outputPos;

  //it will be 2*numElems
  const int numBits = 1;
  const int numBins = 1 << numBits;

  unsigned int* d_bitValueOne;
  checkCudaErrors(cudaMalloc(&d_bitValueOne,sizeof(unsigned int)*numElems));
  unsigned int* d_bitValueZero;
  checkCudaErrors(cudaMalloc(&d_bitValueZero,sizeof(unsigned int)*numElems));
  
  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc(&d_histo,sizeof(unsigned int)*numBins));
  unsigned int* d_pos;
  checkCudaErrors(cudaMalloc(&d_pos,sizeof(unsigned int)*numElems));

  //define the blocks and threads
  size_t threadPerBlock =1024;
  size_t numberOfBlock = numElems/threadPerBlock;
  if(threadPerBlock*numberOfBlock<numElems) numberOfBlock+=1;
 

  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
  //  printf("GPU bit pos %d\n",i);

    checkBit<<<numberOfBlock,threadPerBlock>>>(d_srcVals,d_bitValueOne,d_bitValueZero,i,numElems);
    
    //counts the 0 or 1
//    printf("perform count reduce algo count one\n");
    countReduce(d_bitValueOne,&d_histo[1],numElems);
    
//    printf("perform count reduce algo count zero\n");
    countReduce(d_bitValueZero,&d_histo[0],numElems);
    
    unsigned int h_histo[2];
    checkCudaErrors(cudaMemcpy(h_histo,d_histo,sizeof(unsigned int)*numBins,cudaMemcpyDeviceToHost)); 
    
//    printf("hist bin zero: %d bin one %d\n",h_histo[0],h_histo[1]); 
    
    
    //new position
    //perform scan algorithm
    //and get the new position
    //compact
    //compact for bitValue=0
    scan(d_pos,d_bitValueZero,0,numElems);
    //compact for bitValue=1
    scan(d_pos,d_bitValueOne,h_histo[0],numElems);
   
    /* 
    unsigned int h_pos[20];
    checkCudaErrors(cudaMemcpy(h_pos,d_pos,sizeof(unsigned int)*20,cudaMemcpyDeviceToHost)); 
    for(size_t k=0;k<20;k++){
      printf("copied value index %d new pos %d\n",k,h_pos[k]);
    }
    */

    //map to new position
    gpuMapToPos<<<numberOfBlock,threadPerBlock>>>(d_pos,
		                             	  d_srcVals,d_srcPos,
					     	  d_dstVals,d_dstPos,
					     	  numElems);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
   
//    break;
    std::swap(d_srcVals,d_dstVals);
    std::swap(d_srcPos,d_dstPos);
    
  }
  
  gpuCopy<<<numberOfBlock,threadPerBlock>>>(d_inputVals,d_inputPos,
		                            d_outputVals,d_outputPos,
					    numElems);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());  

  //free memory
  checkCudaErrors(cudaFree(d_bitValueOne));
  checkCudaErrors(cudaFree(d_bitValueZero));
  checkCudaErrors(cudaFree(d_histo));
  checkCudaErrors(cudaFree(d_pos));
}
