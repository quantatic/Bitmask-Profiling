# Bool vs Bitfield Dropout Profiling
The purpose of this repository is to evaluate the performance and memory implications of bool vs bitpacking approaches to storing dropout masks.

## Building and Quick Start

Set the `GPU_CODE` environment variable to specify the CUDA architecture version. Defaults to `sm_80` if no version is passed.

```
$ make clean
rm -f bitmask_read bitmask_write
$ GPU_CODE=sm_80 make
nvcc bitmask_read.cu -o bitmask_read -O3 -g -arch sm_80 -lcurand
nvcc bitmask_write.cu -o bitmask_write -O3 -g -arch sm_80 -lcurand
$ ./bitmask_write bool 1024
computation took 3.019296ms
``` 

### Profiling Invariants

Certain options (such as cuda thread/block counts) are `#define`'d at the top of the CUDA source files. A few invariants must be upheld when running these files.

- Thread size must be evenly divisble by 32 in order to ensure that all warps are fully populated with threads.
- Data size must be evenly divisble by 32 when running bitfield profiling.

## Benchmark List
Benchmarks in this repository include:

- bitmask_read: a workload where the dropout bitmask has already been generated, and is used to actually mask some data.
- bitmask_write: a workload where the dropout mask is created in the kernel, and is wrtten to some output.


## Benchmarking Scripts
Each benchmark includes a profiling harness with which to automatically run and print benchmark results. The mapping from benchmark to profiling harness is listed below.

- bitmask_read: read_profiler.sh
- bitmask_write: write_profiler.sh

A variety of tunable profiling options are present at the top of the profiling harness scripts. Benchmarking results will be printed to stdout in a csv format.

```
$ ./read_profiler.sh 
+ MB=1048576
+ SIZES=(1 2 4 8 16 32 64 128 256 512 1024)
+ NUM_RUNS=1
+ COOLDOWN_TIME=2
+ make clean
rm -f bitmask_read bitmask_write
+ make
nvcc bitmask_read.cu -o bitmask_read -O3 -g -arch sm_80 -lcurand
nvcc bitmask_write.cu -o bitmask_write -O3 -g -arch sm_80 -lcurand
+ BIN_FILENAME=bitmask_read
+ set +x
size,bool,bit
1,computation took 0.056960ms,computation took 0.055072ms
2,computation took 0.101472ms,computation took 0.098464ms
4,computation took 0.265248ms,computation took 0.211744ms
```