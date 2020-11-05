/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
/*
//2020/11/某日作成
やってること
1<<24の要素数でGPUにメモリを確保*3
1個目にはランダムな数字をgpuで計算して格納(kernelA)
2個目と3個目にはそのランダムな数字をもとにexpmodを計算(kernelB)と(kernelC)した結果を格納
最後リダクション総和して4byteの配列に結果を格納(kernelD)

このA→(B,C)→Dのストリームをグラフに見立てて、一括で実行できる
その一括処理をGRAPH_LAUNCH_ITERATIONS回まわすというサンプル
*/


#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#include <cooperative_groups.h>

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS  92

__device__ unsigned int wang_hash(unsigned int seed)
{
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}
//aのb乗mod c
__device__ unsigned int expmod(unsigned int a,unsigned int b,unsigned int c)
{
	unsigned int ans=1;
	unsigned int x=a;
	for(;b>0;b/=2){
		if (b%2==1) ans=ans*x%c;
		x=x*x%c;
	}
	return ans;
}

__global__ void kernelA(unsigned int *inputVec)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int data=wang_hash(tid);
	inputVec[tid]=data;
}

__global__ void kernelB(unsigned int *inputVec,unsigned int *outputVec)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int rnddata=inputVec[tid];
	outputVec[tid]=expmod(2,rnddata,tid+1);
}
__global__ void kernelC(unsigned int *inputVec,unsigned int *outputVec)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int rnddata=inputVec[tid];
	outputVec[tid]=expmod(2,rnddata,tid+2);
}

__global__ void kernelD(unsigned int *outputVecB,unsigned int *outputVecC,unsigned int *result_d)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	atomicAdd(&result_d[0],outputVecB[tid]);
	atomicAdd(&result_d[0],outputVecC[tid]);
}

////////////////////////////////////////////////////////////////////////////////////////////////ここまでGPU処理
////////////////////////////////////////////////////////////////////////////////////////////////ここまでGPU処理

////////////////////////////////////////////////////////////////////////////////////////////////ここからCPU処理
////////////////////////////////////////////////////////////////////////////////////////////////ここからCPU処理












//A→(B,C)→D
void cudaGraphsUsingStreamCapture(unsigned int *inputVec_d, unsigned int *outputVec_dB, unsigned int *outputVec_dC,unsigned int *result_d, int inputSize)
{
	unsigned int result_h;
    cudaStream_t stream1, stream2, streamForGraph;
    cudaEvent_t KernelAEvent,KernelBEvent;
    cudaGraph_t graph;

    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));
    checkCudaErrors(cudaStreamCreate(&streamForGraph));
    checkCudaErrors(cudaEventCreate(&KernelAEvent));
	checkCudaErrors(cudaEventCreate(&KernelBEvent));

	//ここから
    checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

	kernelA<<<inputSize/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d);
	checkCudaErrors(cudaMemsetAsync(outputVec_dB, 0, sizeof(int)*inputSize, stream2));
	checkCudaErrors(cudaMemsetAsync(outputVec_dC, 0, sizeof(int)*inputSize, stream1));
	checkCudaErrors(cudaEventRecord(KernelAEvent, stream1));
	
	checkCudaErrors(cudaStreamWaitEvent(stream2, KernelAEvent, 0));
	kernelB<<<inputSize/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream2>>>(inputVec_d,outputVec_dB);
	checkCudaErrors(cudaEventRecord(KernelBEvent, stream2));
	kernelC<<<inputSize/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d,outputVec_dC);
	checkCudaErrors(cudaStreamWaitEvent(stream1, KernelBEvent, 0));
	
	kernelD<<<inputSize/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_dB,outputVec_dC,result_d);
	checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(int), cudaMemcpyDefault, stream1));//これは最後の1回でいいけどなんとなくいれてる
	
    checkCudaErrors(cudaStreamEndCapture(stream1, &graph));
	//ここまで
	
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    for (int i=0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
       checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
    }
    checkCudaErrors(cudaStreamSynchronize(streamForGraph));

	printf("%d",result_h);

    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));
    checkCudaErrors(cudaStreamDestroy(streamForGraph));
}




int main(int argc, char **argv)
{
    size_t size = 1<<24;    // number of elements to reduce

    // This will pick the best possible CUDA capable device
    int devID = findCudaDevice(argc, (const char **)argv);

    unsigned int *inputVec_d, *outputVec_dB, *outputVec_dC, *result_d;

    checkCudaErrors(cudaMalloc(&inputVec_d, sizeof(int)*size));
    checkCudaErrors(cudaMalloc(&outputVec_dB, sizeof(int)*size));
    checkCudaErrors(cudaMalloc(&outputVec_dC, sizeof(int)*size));
	checkCudaErrors(cudaMalloc(&result_d, sizeof(int)));
	
    cudaGraphsUsingStreamCapture(inputVec_d, outputVec_dB, outputVec_dC,result_d, size);
	
    checkCudaErrors(cudaFree(inputVec_d));
    checkCudaErrors(cudaFree(outputVec_dB));
    checkCudaErrors(cudaFree(outputVec_dC));
	checkCudaErrors(cudaFree(result_d));
    return EXIT_SUCCESS;
}
