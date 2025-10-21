#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

cudaError_t sortWithCuda(int* result, const int* array, unsigned int size);


__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}


__global__ void bitonicSortKernel(int* array, int j, int k, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n) return;

    int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
        
            if (array[i] > array[ixj]) {
                swap(&array[i], &array[ixj]);
            }
        }
        else {
       
            if (array[i] < array[ixj]) {
                swap(&array[i], &array[ixj]);
            }
        }
    }
}



int main() {
    const unsigned int arraySize = 1000000;
    int* array = (int*)malloc(arraySize * sizeof(int));
    int* sorted_array = (int*)malloc(arraySize * sizeof(int));
    int result = 0;


    srand((unsigned int)time(NULL));
    for (int i = 0; i < arraySize; i++) {
        array[i] = rand() % 100000;
    }

    printf("Array size: %d elements\n", arraySize);

   

    cudaError_t cudaStatus = sortWithCuda(sorted_array, array, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sortWithCuda failed!");
        free(array);
        free(sorted_array);
        return 1;
    }

 

 

    free(array);
    free(sorted_array);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t sortWithCuda(int* result, const int* array, unsigned int size) {
    int* dev_array = 0;
    int* dev_result_check = 0;
    cudaError_t cudaStatus;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

 
    int n = 1;
    while (n < size) {
        n <<= 1;
    }
 

    int blockSize = 128;
    int numBlocks = (n + blockSize - 1) / blockSize;
    std::cout << "Nubmer of blocks: " << numBlocks << std::endl;

 
    cudaStatus = cudaMalloc((void**)&dev_array, n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_array!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_result_check, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_result_check!");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_array, 0, n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_array, array, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(start);


    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel << <numBlocks, blockSize >> > (dev_array, j, k, n);

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error;
            }

            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d!\n", cudaStatus);
                goto Error;
            }
        }
    }




    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code after check!\n");
        goto Error;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);


    cudaStatus = cudaMemcpy(result, dev_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    int check_result;
    cudaStatus = cudaMemcpy(&check_result, dev_result_check, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for result check failed!");
        goto Error;
    }

    printf("Time taken: %.6f seconds\n", gpu_time / 1000.0f);


Error:
    cudaFree(dev_array);
    cudaFree(dev_result_check);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaStatus;
}
