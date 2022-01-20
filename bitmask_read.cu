#include "cuda_runtime.h"

#include <stdlib.h>
#include <stdio.h>

#define MASK_SIZE (sizeof(int) * 8)

#define NUM_THREADS 128
#define NUM_BLOCKS 128

__global__ void maskBool(const int *input, const bool *mask, int *output, int n) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < n; i += stride) {
		if (mask[i]) {
			output[i] = input[i];
		} else {
			output[i] = 0;
		}
	}
}

__global__ void maskBit(const int *input, const int *bitMask, int *output, int n) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < n; i += stride) {
		int bitIndex = i % MASK_SIZE;
		int maskIndex = i / MASK_SIZE;

		bool used = (bitMask[maskIndex] & (1 << bitIndex)) != 0;
		if (used) {
			output[i] = input[i];
		} else {
			output[i] = 0;
		}
	}
}

// prints usage and exits with code -1
void usage(char *programName) {
	fprintf(stderr, "usage: %s <bool | bit> <num-elements>\n", programName);
	exit(-1);
}


int main(int argc, char **argv) {
	char *programName = argv[0];
	if (argc != 3) {
		usage(programName);
	}

	bool useBit;
	char *runType = argv[1];
	if(strcmp(runType, "bool") == 0) {
		useBit = false;
	} else if (strcmp(runType, "bit") == 0) {
		useBit = true;
	} else {
		usage(programName);
	}

	int numElements = atoi(argv[2]);

	int *input = new int[numElements];

	bool *mask;
	int *bitMask;
	if(useBit) {
		bitMask = new int[numElements / MASK_SIZE];
	} else {
		mask = new bool[numElements];
	}


	int *expectedOutput = new int[numElements];

	srand(1304);
	for(int i = 0; i < numElements; i++) {
		int value = rand();
		input[i] = value;
		if (rand() > (RAND_MAX / 2)) {
			expectedOutput[i] = value;
		
			if(useBit) {
				int oldMask = bitMask[i / MASK_SIZE];
				int newMask = oldMask | (1 << (i % MASK_SIZE));
				bitMask[i / MASK_SIZE] = newMask;
			} else {
				mask[i] = true;
			}

		} else {
			if(!useBit) {
				mask[i] = false;
			}

			expectedOutput[i] = 0;
		}
	}

	int *deviceBitMask;
	bool *deviceMask;
	if(useBit) {
		cudaMalloc(&deviceBitMask, sizeof(int) * numElements / MASK_SIZE);
		cudaMemcpy(deviceBitMask, bitMask, sizeof(int) * numElements / MASK_SIZE, cudaMemcpyHostToDevice);
	} else {
		cudaMalloc(&deviceMask, numElements * sizeof(bool));
		cudaMemcpy(deviceMask, mask, numElements * sizeof(bool), cudaMemcpyHostToDevice);
	}

	int *deviceInput = nullptr;
	int *deviceOutput = nullptr;

	cudaMalloc(&deviceInput, numElements * sizeof(int));
	cudaMalloc(&deviceOutput, numElements * sizeof(int));

	cudaMemcpy(deviceInput, input, numElements * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	if(useBit) {
		maskBit<<<NUM_BLOCKS, NUM_THREADS>>>(deviceInput, deviceBitMask, deviceOutput, numElements);
	} else {
		maskBool<<<NUM_BLOCKS, NUM_THREADS>>>(deviceInput, deviceMask, deviceOutput, numElements);
	}
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	int *actualOutput = new int[numElements];
	cudaMemcpy(actualOutput, deviceOutput, numElements * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < numElements; i++) {
		if(expectedOutput[i] != actualOutput[i]) {
			fprintf(stderr, "index %d, expected %d, got %d\n", i, expectedOutput[i], actualOutput[i]);
			exit(-1);
		}
	}

	float msElapsed;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msElapsed, start, stop);

	printf("computation took %fms\n", msElapsed);
}