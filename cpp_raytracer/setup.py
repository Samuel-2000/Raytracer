# setup.py
from setuptools import setup, Extension
import pybind11
import os
import sys

def get_optimization_flags():
    flags = []
    
    if os.name == 'nt':  # Windows
        flags = [
            '/O2', '/Ob2', '/Oi', '/Ot', '/Oy', '/GT', '/GL',
            '/std:c++17', '/openmp:llvm', '/fp:fast', '/arch:AVX2',
        ]
    else:  # Linux/Mac
        flags = [
            '-O3', '-march=native', '-ffast-math', '-fopenmp',
            '-funroll-loops', '-ftree-vectorize', '-fno-trapping-math',
            '-std=c++17', '-fopenmp-simd', '-mtune=native',
            '-fomit-frame-pointer',
        ]
        
        import platform
        if platform.machine() in ['x86_64', 'amd64']:
            flags.extend(['-msse4.2', '-mavx', '-mfma'])
    
    return flags

if '--debug' in sys.argv:
    optimization_flags = ['-O0', '-g'] if os.name != 'nt' else ['/Od', '/Zi']
    sys.argv.remove('--debug')
else:
    optimization_flags = get_optimization_flags()

ext_modules = [
    Extension(
        "raytracer_cpp",
        [
            "binding.cpp",
            "raytracer_core.cpp", 
            "bvh.cpp",
            "textures.cpp"
        ],
        include_dirs=[".", pybind11.get_include()],
        language='c++',
        extra_compile_args=optimization_flags,
        extra_link_args=['/openmp'] if os.name == 'nt' else ['-fopenmp'],
    ),
]

setup(
    name="raytracer_cpp",
    version="1.0.0",
    description="Optimized Ray Tracer with Textures and Skybox",
    ext_modules=ext_modules,
    zip_safe=False,
)