#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


// __global__ void scatter_max_cuda_kernel(
//   const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,
//   const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> input_index,
//   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output,
//   int64_t numel)
// {

//   const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
//   if(index < numel)
//   {
//     const int32_t row = index / input.size(1);
//     const int32_t col = index - row * input.size(1);

//     const int32_t row_output = input_index[row][col];  
//     atomicMaxFloat(&output[row_output][col], input[row][col]);
//   }
// }

// __global__ void get_index_cuda_kernel(
//   const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,
//   const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> input_index,
//   const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output,
//   torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> output_index,
//   int64_t numel)
// {
//   const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
//   if(index < numel)
//   {
//     const int32_t row = index / input.size(1);
//     const int32_t col = index - row * input.size(1);
    
//     const int32_t row_output = input_index[row][col];  
//     if(output[row_output][col] == input[row][col])
//     {
//       output_index[row_output][col] = row;
//     }
//   }
// }

// // input float32
// // input_index int64
// // output float32
// // output_index int64

// void scatter_max_cuda(torch::Tensor input, torch::Tensor input_index,
//                       torch::Tensor output, torch::Tensor output_index,
//                       bool train)
// {
//   cudaSetDevice(input.get_device());

//   int32_t threads = 1024;
//   int64_t blocks = (input.numel() + threads - 1) / threads;

//   scatter_max_cuda_kernel<<<blocks, threads>>>(
//     input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//     input_index.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
//     output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//     input.numel());

//   if(train)
//   {
//     get_index_cuda_kernel<<<blocks, threads>>>(
//       input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//       input_index.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
//       output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//       output_index.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
//       input.numel());
//   }
// }

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
  int64_t size = input.size(0) * chw;
 

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









