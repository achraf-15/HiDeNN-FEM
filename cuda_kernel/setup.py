from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_kernel',
    packages=['cuda_kernel'],
    ext_modules=[
        CUDAExtension(
            name='cuda_kernel.invert_cuda',
            sources=['cuda_kernel/invert_kernel.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            name='cuda_kernel.solve_cuda',
            sources=['cuda_kernel/solve_kernel.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O2']}
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
