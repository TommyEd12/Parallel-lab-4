#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

cudaError_t sumWithCuda(int* result, const int* array, unsigned int size);

__global__ void sumKernel(int* partial_sums, const int* array, int n)
{
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;


    shared_data[tid] = (i < n) ? array[i] : 0;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

 
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

__global__ void finalSumKernel(int* result, const int* partial_sums, int n)
{
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;


    shared_data[tid] = (tid < n) ? partial_sums[tid] : 0;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = shared_data[0];
    }
}

int main()
{
    const unsigned int arraySize = 1000000; 
    int* array = (int*)malloc(arraySize * sizeof(int));
    int result = 0;


    srand((unsigned int)time(NULL));
    for (int i = 0; i < arraySize; i++) {
        array[i] = rand() % 1000; 
    }

    
    

    printf("Array size: %d elements\n", arraySize);


    cudaError_t cudaStatus = sumWithCuda(&result, array, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sumWithCuda failed!");
        free(array);
        return 1;
    }

    printf("Array sum: %d\n", result);
 

    free(array);


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


cudaError_t sumWithCuda(int* result, const int* array, unsigned int size)
{
    int* dev_array = 0;
    int* dev_partial_sums = 0;
    int* dev_result = 0;
    cudaError_t cudaStatus;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    int blockSize = 128;
    int numBlocks = (size + blockSize - 1) / blockSize;
    std::cout << "Number of blocks: " << numBlocks << std::endl;

  
    cudaStatus = cudaMalloc((void**)&dev_array, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_array!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_partial_sums, numBlocks * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_partial_sums!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_result, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_result!");
        goto Error;
    }

  
    cudaStatus = cudaMemcpy(dev_array, array, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaEventRecord(start);
    for (int i = 0; i < 10000; i++) {

        sumKernel <<<numBlocks, blockSize, blockSize * sizeof(int) >> > (dev_partial_sums, dev_array, size);

 
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "sumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }


        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after first kernel!\n", cudaStatus);
            goto Error;
        }


        if (numBlocks > 1) {
            finalSumKernel <<<1, 256, 256 * sizeof(int) >> > (dev_result, dev_partial_sums, numBlocks);
          
        }
        else {
 
            cudaStatus = cudaMemcpy(dev_result, dev_partial_sums, sizeof(int), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy DeviceToDevice failed!");
                goto Error;
            }
        }


        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "finalSumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }


        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after second kernel!\n", cudaStatus);
            goto Error;
        }
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);


    cudaStatus = cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
 
    printf("Time taken: %.6f seconds\n", gpu_time / 1000.0f);

Error:
    cudaFree(dev_array);
    cudaFree(dev_partial_sums);
    cudaFree(dev_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaStatus;
}
