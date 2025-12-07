# setup_recurrent.py
"""
Build script for the recurrent Mamba step kernel extension.

Usage:
    # CPU only (no CUDA)
    python setup_recurrent.py build_ext --inplace
    
    # With CUDA
    python setup_recurrent.py build_ext --inplace --cuda

This builds the recurrent_mamba_step_ext module that provides O(T) streaming.
"""

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Check for CUDA flag
USE_CUDA = '--cuda' in sys.argv
if USE_CUDA:
    sys.argv.remove('--cuda')

# Source files
this_dir = Path(__file__).parent
csrc_dir = this_dir / 'mamba_ssm' / 'ops' / 'csrc'

sources = [
    str(csrc_dir / 'recurrent_mamba_step.cpp'),
]

if USE_CUDA:
    sources.append(str(csrc_dir / 'recurrent_mamba_step_cuda.cu'))

# Define macros
define_macros = []
if USE_CUDA:
    define_macros.append(('WITH_CUDA', None))

# Extension class
ExtClass = CUDAExtension if USE_CUDA else CppExtension

ext_modules = [
    ExtClass(
        name='mamba_ssm.ops.recurrent_mamba_step_ext',
        sources=sources,
        define_macros=define_macros,
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '--use_fast_math'] if USE_CUDA else [],
        },
    ),
]

setup(
    name='recurrent_mamba_step',
    version='0.1.0',
    author='MMABA-PSEUDO',
    description='Recurrent Mamba step kernel for O(T) streaming',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
)
