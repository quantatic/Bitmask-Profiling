GPU_CODE ?= sm_80
NVCC := nvcc
CUDAFLAGS := -O3 -g -arch $(GPU_CODE) -lcurand

SRC_FILES := $(wildcard *.cu)
BIN_FILES := $(patsubst %.cu,%,$(SRC_FILES))

all: $(BIN_FILES)

%: %.cu
	$(NVCC) $^ -o $@ $(CUDAFLAGS)

clean:
	rm -f $(BIN_FILES)

.phony: clean
