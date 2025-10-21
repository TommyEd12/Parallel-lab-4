#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

#define VECTOR_SIZE 100000
#define NUM_TRIALS 100


__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < VECTOR_SIZE) c[i] = a[i] + b[i];
}

__global__ void subKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < VECTOR_SIZE) c[i] = a[i] - b[i];
}

__global__ void mulKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < VECTOR_SIZE) c[i] = a[i] * b[i];
}

__global__ void divKernel(float* c, const float* a, const float* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < VECTOR_SIZE && b[i] != 0) c[i] = a[i] / b[i];
}


#include <chrono>
#include <iostream>



int main() {
    setlocale(LC_ALL, "rus");

    int* a = (int*)malloc(VECTOR_SIZE * sizeof(int));
    int* b = (int*)malloc(VECTOR_SIZE * sizeof(int));
    int* c = (int*)malloc(VECTOR_SIZE * sizeof(int));
    float* a_float = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* b_float = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* c_float = (float*)malloc(VECTOR_SIZE * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = rand() % 1000;
        b[i] = rand() % 1000 + 1;
        a_float[i] = (float)a[i];
        b_float[i] = (float)b[i];
    }

  
    int* dev_a, * dev_b, * dev_c;
    float* dev_a_f, * dev_b_f, * dev_c_f;


    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, VECTOR_SIZE * sizeof(int));


    cudaMalloc((void**)&dev_a_f, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_b_f, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_c_f, VECTOR_SIZE * sizeof(float));


    cudaMemcpy(dev_a, a, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a_f, a_float, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b_f, b_float, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);


    int blockSize = 128;
    int numBlocks = (VECTOR_SIZE + blockSize - 1) / blockSize;
    std::cout << "Число блоков: " << numBlocks << std::endl;


    auto runAddInt = [&]() {
        addKernel <<<numBlocks, blockSize >> > (dev_c, dev_a, dev_b);
        cudaDeviceSynchronize();
        };

    auto runSubInt = [&]() {
        subKernel <<<numBlocks, blockSize >> > (dev_c, dev_a, dev_b);
        cudaDeviceSynchronize();
        };

    auto runMulInt = [&]() {
        mulKernel <<<numBlocks, blockSize >> > (dev_c, dev_a, dev_b);
        cudaDeviceSynchronize();
        };

    auto runDivFloat = [&]() {
        divKernel <<<numBlocks, blockSize >> > (dev_c_f, dev_a_f, dev_b_f);
        cudaDeviceSynchronize();
        };

   
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_TRIALS; i++) {
        runAddInt();
        runSubInt();
        runMulInt();
        runDivFloat();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> overallTime = end - start;
       


    cudaMemcpy(c, dev_c, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_float, dev_c_f, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_a_f);
    cudaFree(dev_b_f);
    cudaFree(dev_c_f);

    free(a);
    free(b);
    free(c);
    free(a_float);
    free(b_float);
    free(c_float);

  
    std::cout << "Время выолнения 100 запусков" << std::endl;
    std::cout << "" << overallTime.count() <<" сек." << std::endl;

    return 0;
}
