# ================================================
# FILE: cpp_raytracer/setup.py (OPTIMIZED)
# ================================================
from setuptools import setup, Extension
import pybind11
import os
import sys

# Determine compiler flags based on platform and CPU
def get_optimization_flags():
    flags = []
    
    if os.name == 'nt':  # Windows
        flags = [
            '/O2',  # Maximum optimization
            '/Ob2',  # Aggressive inlining
            '/Oi',   # Enable intrinsic functions
            '/Ot',   # Favor fast code
            '/Oy',   # Omit frame pointers
            '/GT',   # Support thread-local storage
            '/GL',   # Whole program optimization
            '/std:c++17',
            '/openmp:llvm',
            '/fp:fast',  # Fast floating point
            '/arch:AVX2',  # Enable AVX2 if available
        ]
    else:  # Linux/Mac
        flags = [
            '-O3',  # Maximum optimization
            '-march=native',  # Use CPU-specific instructions
            '-ffast-math',  # Fast math (less precise but faster)
            '-fopenmp',  # OpenMP support
            '-funroll-loops',  # Unroll loops
            '-ftree-vectorize',  # Vectorize loops
            '-fno-trapping-math',  # Assume no floating point traps
            '-std=c++17',
            '-fopenmp-simd',  # OpenMP SIMD directives
            '-mtune=native',  # Tune for native CPU
            '-fomit-frame-pointer',  # Omit frame pointers
        ]
        
        # Add CPU-specific SIMD flags
        import platform
        if platform.machine() in ['x86_64', 'amd64']:
            flags.extend(['-msse4.2', '-mavx', '-mfma'])
    
    return flags

# Check if user wants to disable optimizations
if '--debug' in sys.argv:
    optimization_flags = ['-O0', '-g'] if os.name != 'nt' else ['/Od', '/Zi']
    sys.argv.remove('--debug')
else:
    optimization_flags = get_optimization_flags()

ext_modules = [
    Extension(
        "raytracer_cpp",  # SAME MODULE NAME
        [
            "binding.cpp",
            "raytracer_core.cpp", 
            "bvh.cpp"
        ],
        include_dirs=[".", pybind11.get_include()],
        language='c++',
        extra_compile_args=optimization_flags,
        extra_link_args=['/openmp'] if os.name == 'nt' else ['-fopenmp'],
    ),
]

setup(
    name="raytracer_cpp",  # SAME PACKAGE NAME
    version="1.0.0",
    description="Optimized Ray Tracer with OpenMP and SIMD",
    ext_modules=ext_modules,
    zip_safe=False,
)