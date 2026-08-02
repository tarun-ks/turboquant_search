import os
import sys
from setuptools import setup, find_packages

# Try to build C++ extension; fall back gracefully if pybind11 is missing
_ext_modules = []
_cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext = Pybind11Extension(
        "turboquant_search._tqs_cpp",
        sources=[
            "csrc/tqs_kernels.cpp",
            "csrc/bindings.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
    )

    if sys.platform == "darwin":
        import platform as _plat
        # On Apple Silicon, -march=native emits the exact CPU (e.g. apple-m4),
        # which older clang toolchains do not recognise. -mcpu=apple-m1 is the
        # ISA baseline shared by all M-series and builds portably; on Intel
        # macOS keep -march=native.
        _arch_flag = "-mcpu=apple-m1" if _plat.machine() == "arm64" else "-march=native"
        ext.extra_compile_args += ["-O3", "-ffast-math", _arch_flag]
        ext.extra_link_args += ["-framework", "Accelerate"]
        # OpenMP: skip on macOS due to compatibility issues with Python
        # Parallelization is done at the Python level instead
    elif sys.platform == "linux":
        ext.extra_compile_args += ["-O3", "-ffast-math", "-march=native", "-fopenmp"]
        ext.extra_link_args += ["-fopenmp"]
    else:
        ext.extra_compile_args += ["-O2"]

    _ext_modules = [ext]
    _cmdclass = {"build_ext": build_ext}

except ImportError:
    pass

# Allow env var to force skip C++ build
if os.environ.get("TQS_NO_NATIVE", "0") == "1":
    _ext_modules = []
    _cmdclass = {}

setup(
    name="turboquant-search",
    version="0.3.0",
    author="Tarun",
    description="Vector compression for similarity search — IVF-TQ: 88% recall on SIFT-1M at 6x compression, zero codebook training.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tarun-ks/turboquant_search",
    packages=find_packages(),
    ext_modules=_ext_modules,
    cmdclass=_cmdclass,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "native": ["pybind11>=2.11.0"],
        "faiss": ["faiss-cpu>=1.7.0"],
        "all": ["faiss-cpu>=1.7.0", "datasets>=2.0.0", "pybind11>=2.11.0", "matplotlib>=3.7.0"],
    },
    entry_points={
        "console_scripts": [
            "tqs=turboquant_search.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
