from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scatter_index',
    description="mvf paper maxpooling function",
    author='zilong Yuan',
    ext_modules=[
        CUDAExtension('scatter_index', [
            'cuda/scatter_index.cpp',
            'cuda/scatter_index_cuda.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })