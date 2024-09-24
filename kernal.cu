#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ 
void cuda_hello() {
	printf("Hello World from GPU!\n");
}


__global__
void add(float* A, float* B, float* C) {

	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}


int main() {

	/*
	int N = 3;
	float A[3] = {2.0f, 2.0f, 2.0f};
	float B[3] = {2.0f, 2.0f, 2.0f};
	float C[3];
	*/

	cuda_hello <<< 1, 1 >>> ();
	return 0;
}

