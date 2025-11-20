from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='invert_cuda',
    packages=['cuda_kernel'],
    ext_modules=[
        CUDAExtension(
            name='cuda_kernel.invert_cuda',
            sources=['cuda_kernel/invert_kernel.cu'],  
            extra_compile_args={'cxx': [], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
