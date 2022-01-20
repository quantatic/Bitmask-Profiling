#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define FULL_MASK 0XFFFFFFFF

#define MASK_SIZE (sizeof(int) * 8)
#define SET_BIT(value, index) ((value) | (1 << (index)))
#define CLEAR_BIT(value, index) ((value) & (~(1 << (index))))
#define GET_BIT(value, index) (((value) & (1 << (index))) != 0)

#define NUM_THREADS 128
#define NUM_BLOCKS 128

#define MASK_THRESHOLD 0.5

#define MASK_SEED 1234
#define VALUE_SEED 1304

__global__ void cudaRand(const int n, float *output) {
    unsigned long long idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    curandStateXORWOW_t state;
    curand_init(MASK_SEED, idx, 0, &state);

	for(int i = idx; i < n; i += stride) {
		output[i] = curand_uniform(&state);
	}
}

__global__ void dropoutBool(const int n, const int *input, bool *mask, int *output) {
    unsigned long long idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    curandStateXORWOW_t state;
    curand_init(MASK_SEED, idx, 0, &state);

    for(int i = idx; i < n; i += stride) {
    	bool masked = curand_uniform(&state) > MASK_THRESHOLD;
		mask[i] = masked;
		if (masked) {
			output[i] = input[i];
		} else {
			output[i] = 0;
		}
    }
}

// must be run with NUM_THREADS % 32 == 0
__global__ void dropoutBit(const int n, const int *input, unsigned *bitMask, int *output) {
	// assert(__activemask() == FULL_MASK);

    unsigned long long idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int laneId = threadIdx.x % 32;

    curandStateXORWOW_t state;
    curand_init(MASK_SEED, idx, 0, &state);

    for(int i = idx; i < n; i += stride) {
    	bool masked = curand_uniform(&state) > MASK_THRESHOLD;
		unsigned warpBitMask = __ballot_sync(FULL_MASK, masked);
		// unsigned warpBitMask = __reduce_or_sync(FULL_MASK, masked ? SET_BIT(0, laneId) : 0);
		int bitMaskIndex = i / MASK_SIZE;

		// write result out using only a single lane
		if(laneId == 0) {
			bitMask[bitMaskIndex] = warpBitMask;
		}

		if (masked) {
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

	long numElements = atol(argv[2]);
	// number of elements must be able to be cleanly packed into an int
	assert(numElements % MASK_SIZE == 0);

	int *input = new int[numElements];

	srand(VALUE_SEED);
	for(int i = 0; i < numElements; i++) {
		int value = rand();
		input[i] = value;
	}

	bool *deviceMask = nullptr;
	unsigned *deviceBitMask = nullptr;
	int *deviceInput;
	int *deviceOutput;
	float *deviceRandValues;

	if(useBit) {
		CUDA_CALL(cudaMalloc(&deviceBitMask, numElements * sizeof(unsigned) / MASK_SIZE));
	} else {
		CUDA_CALL(cudaMalloc(&deviceMask, numElements * sizeof(bool)));
	}
	CUDA_CALL(cudaMalloc(&deviceInput, numElements * sizeof(int)));
	CUDA_CALL(cudaMalloc(&deviceOutput, numElements * sizeof(int)));
	CUDA_CALL(cudaMalloc(&deviceRandValues, numElements * sizeof(float)));

	CUDA_CALL(cudaMemcpy(deviceInput, input, numElements * sizeof(int), cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));

	CUDA_CALL(cudaEventRecord(start));
	if(useBit) {
		dropoutBit<<<NUM_BLOCKS, NUM_THREADS>>>(numElements, deviceInput, deviceBitMask, deviceOutput);
	} else {
		dropoutBool<<<NUM_BLOCKS, NUM_THREADS>>>(numElements, deviceInput, deviceMask, deviceOutput);
	}
	CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaDeviceSynchronize());

    cudaRand<<<NUM_BLOCKS, NUM_THREADS>>>(numElements, deviceRandValues);
    CUDA_CALL(cudaDeviceSynchronize());

    bool *mask = nullptr;
	unsigned *bitMask = nullptr;
	int *output = new int[numElements];
	float *randValues = new float[numElements];

	if(useBit) {
		bitMask = new unsigned[numElements / MASK_SIZE];
		CUDA_CALL(cudaMemcpy(bitMask, deviceBitMask, numElements * sizeof(unsigned) / MASK_SIZE, cudaMemcpyDeviceToHost));
	} else {
		mask = new bool[numElements];
		CUDA_CALL(cudaMemcpy(mask, deviceMask, numElements * sizeof(bool), cudaMemcpyDeviceToHost));
	}
	CUDA_CALL(cudaMemcpy(output, deviceOutput, numElements * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(randValues, deviceRandValues, numElements * sizeof(float), cudaMemcpyDeviceToHost));

	bool *expectedMask = new bool[numElements];
	unsigned *expectedBitMask = new unsigned[numElements / MASK_SIZE];
	int *expectedOutput = new int[numElements];

	for(int i = 0; i < numElements; i++) {
		int bitmaskValueIndex = i / MASK_SIZE;
		int bitmaskBitIndex = i % MASK_SIZE;

		unsigned &bitmaskValue = expectedBitMask[bitmaskValueIndex];

		// Ensure bitmask value is initialized
		if(bitmaskBitIndex == 0) {
			bitmaskValue = 0;
		}

		if(randValues[i] > MASK_THRESHOLD) {
			expectedMask[i] = true;
			expectedOutput[i] = input[i];
			bitmaskValue = SET_BIT(bitmaskValue, bitmaskBitIndex);
		} else {
			expectedMask[i] = false;
			expectedOutput[i] = 0.0f;
			bitmaskValue = CLEAR_BIT(bitmaskValue, bitmaskBitIndex);
		}
	}

	// check elements results
	for(int i = 0; i < numElements; i++) {
		assert(expectedOutput[i] == output[i]);
	}

	// check mask/bitmask results
	if(useBit) {
		for(int i = 0; i < numElements / MASK_SIZE; i++) {
			assert(expectedBitMask[i] == bitMask[i]);
		}
	} else {
		for(int i = 0; i < numElements; i++) {
			assert(expectedMask[i] == mask[i]);
		}
	}

	float msElapsed;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msElapsed, start, stop);

	printf("computation took %fms\n", msElapsed);
}