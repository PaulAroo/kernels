/*

- equation to convert each colored pixel to its grayscale counterpart:
  -> L = 0.21*r + 0.72*g + 0.07*b    ((luminance formula)

  dimension/size of input: width * height * number_of_channels
  dimension/size of output: width * height

  p_out -> output pixels (converted to grayscale) [l1, l2, ......]
  p_in -> input pixels (coloured image) [r1, g1, b1, r2, g2, b2, ......]
  *_h -> allocated on host
  *_d -> allocated on device

*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <string>
#include <math.h>
#include <cmath>
#include <cstdlib> // for rand() and srand()
// #include <ctime>    // for time()
#include <vector>

template<class F>
long benchmark_one_ms(F&& f) {
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

#define NUM_CHANNELS 3
#define NUM_THREADS_PER_DIM 16

__global__ void grayscaleConversionKernel(int* p_in_d, int* p_out_d, int width, int height) {

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  // this is slower (mem access isn't coalesced?)
  // const size_t col = blockIdx.y * blockDim.y + threadIdx.y;
  // const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

  // Assume row-major form of storage
  if(col < width && row < height) {
    const size_t pixelOffset = row * width + col;
    const size_t rgbOffset = pixelOffset * NUM_CHANNELS;

    int r = p_in_d[rgbOffset];
    int g = p_in_d[rgbOffset + 1];
    int b = p_in_d[rgbOffset + 2];

    float luminance_val = 0.21f * r + 0.72f * g + 0.07f * b;
    p_out_d[pixelOffset] = static_cast<int>(roundf(luminance_val));
  }
}

void compareSequentialAndParallelResults(std::vector<int> parr, std::vector<int> seq, int size) {
  for(size_t i = 0; i < size; i++) {
    if(parr[i] != seq[i]) {
      std::cout << "Error: results do not match at index " << i << ", " << parr[i] << " is not equal to " << seq[i] << "\n";
      exit(1);
    }
  }
}

void computeSequentially(std::vector<int> p_in, std::vector<int> p_out, int size) {

  for (size_t i = 0; i < size; i++) {
    const size_t rgbOffset = i * NUM_CHANNELS;

    int r = p_in[rgbOffset];
    int g = p_in[rgbOffset + 1];
    int b = p_in[rgbOffset + 2];

    float luminance_val = 0.21f * r + 0.72f * g + 0.07f * b;
    p_out[i] = static_cast<int>(std::round(luminance_val));

  }
}

void initializeData(std::vector<int> p_in_h, size_t size) {
  // srand(time(NULL)); randomized
  srand(42); // Seed the random number generator

  for(size_t i = 0; i < size; i++) {
    p_in_h[i] = rand() % 256; // generate random numbers between 0 and 255
  }
}


int main(int argc, char** argv) {

  // int device;
  // cudaGetDevice(&device);

  // cudaDeviceProp props;
  // cudaGetDeviceProperties(&props, device);

  // std::cout << "Device: " << props.name << "\n";
  // std::cout << "Compute capability: " << props.major << "." << props.minor << "\n";

  if(argc < 3) {
    printf("usage: ./grayscale.out <WIDTH> <HEIGHT> \n");
    exit(1);
  }

  const int width = std::stoi(argv[1]);
  const int height = std::stoi(argv[2]);

  const size_t size = width * height * NUM_CHANNELS;
  const size_t reduced_size = width * height;

  std::vector<int> p_in_h(size);
  std::vector<int> p_out_h(reduced_size);
  
  initializeData(p_in_h, size);

  int* p_in_d;
  int* p_out_d;

  cudaMalloc((void **) &p_in_d, size * sizeof(int));
  cudaMalloc((void **) &p_out_d, reduced_size * sizeof(int));

  cudaMemcpy(p_in_d, p_in_h.data(), size * sizeof(int), cudaMemcpyHostToDevice);

  const dim3 numBlocks(
    (width + NUM_THREADS_PER_DIM - 1) / NUM_THREADS_PER_DIM,
    (height + NUM_THREADS_PER_DIM - 1) / NUM_THREADS_PER_DIM,
    1
  );
  const dim3 numThreads(NUM_THREADS_PER_DIM, NUM_THREADS_PER_DIM, 1);

  long gpu_time_ms = benchmark_one_ms([&]() {
    grayscaleConversionKernel<<<numBlocks, numThreads>>>(p_in_d, p_out_d, width, height);
    cudaDeviceSynchronize();
  });
  std::cout << "GPU: " << gpu_time_ms << " ms" << std::endl;
  
  cudaMemcpy(p_out_h.data(), p_out_d, reduced_size * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<int> p_out(reduced_size);

  long cpu_time_ms = benchmark_one_ms([&]() {
    computeSequentially(p_in_h, p_out, reduced_size);
  });
  std::cout << "CPU: " << cpu_time_ms << " ms" << std::endl;

  compareSequentialAndParallelResults(p_out_h, p_out, reduced_size);


  cudaFree(p_in_d);
  cudaFree(p_out_d);

  return 0;
}