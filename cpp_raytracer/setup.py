from setuptools import setup, Extension
import pybind11
import os
import sys

# Remove problematic flags
if os.name == 'nt':  # Windows
    compile_args = [
        '/O2',           # Maximum optimization
        '/Ob2',          # Aggressive inlining
        '/Oi',           # Enable intrinsic functions
        '/Ot',           # Favor fast code
        '/std:c++17',
        '/openmp',
        '/fp:fast',      # Fast floating point
        '/arch:AVX2',    # Enable AVX2
    ]
    link_args = ['/openmp']
else:  # Linux/Mac
    compile_args = [
        '-O3',          # Maximum optimization
        '-march=native', # Use CPU-specific instructions
        '-ffast-math',  # Fast math
        '-fopenmp',     # OpenMP support
        '-funroll-loops', # Unroll loops
        '-std=c++17',
        '-mavx2',       # AVX2 instructions
        '-mfma',        # FMA instructions
    ]
    link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "raytracer_cpp",
        sources=["raytracer_core.cpp"],
        include_dirs=[".", pybind11.get_include()],
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name="raytracer_cpp",
    version="2.0.0",
    description="High-Performance Ray Tracer with AVX2 and OpenMP",
    ext_modules=ext_modules,
    zip_safe=False,
)