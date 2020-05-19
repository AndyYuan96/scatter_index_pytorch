#include <torch/extension.h>
#include <iostream>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be CUDA tensor")


void scatter_index_cuda(torch::Tensor input, torch::Tensor index, torch::Tensor output);

void scatter_index(torch::Tensor input, torch::Tensor index, torch::Tensor output)
{
  CHECK_CUDA(input);
  CHECK_CUDA(index);
  CHECK_CUDA(output);

  scatter_index_cuda(input, index, output);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_index", &scatter_index, "scatter_index");
}
