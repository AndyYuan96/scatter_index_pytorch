#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void scatter_index_cuda_kernel(
  const scalar_t* input,
  const long* index,
  scalar_t* output,
  int64_t c, int64_t hw, int64_t chw, int64_t size)
{
    const int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size)
    {
      const int32_t b = idx / hw;
      const int32_t t = idx - b * hw;
      output[idx] = input[b * chw + index[idx] * hw + t]; 
    }
}


void scatter_index_cuda(torch::Tensor input, torch::Tensor index, torch::Tensor output)
{
  cudaSetDevice(input.get_device());
  // input : (B, C, H, W)
  // index : (B, H, W)
  // output : (B, H, W)

  auto dim_i = input.dim();
  auto index_i = index.dim();
  assert (dim_i == 4);
  assert (index_i == 3);

  int64_t c = input.size(1);
  int64_t hw = input.size(2) * input.size(3);
  int64_t chw = c * hw;
  int64_t size = input.size(0) * hw;
 

  int64_t threads = 1024;
  int64_t blocks = (input.numel() + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.type(), "scatter_index_cuda", ([&]{
    scatter_index_cuda_kernel<scalar_t><<<blocks, threads>>>(
      input.data<scalar_t>(),
      index.data<long>(),
      output.data<scalar_t>(),
      c, hw, chw, size
    );
  }));
}









