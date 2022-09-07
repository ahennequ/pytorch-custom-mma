import sys
from functools import lru_cache
from subprocess import DEVNULL, call
from setuptools import setup, find_packages

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

@lru_cache(None)
def cuda_toolkit_available():
  try:
    call(["nvcc"], stdout = DEVNULL, stderr = DEVNULL)
    return True
  except FileNotFoundError:
    return False

def compile_args():
  args = ["-fopenmp", "-ffast-math"]
  if sys.platform == "darwin":
    args = ["-Xpreprocessor", *args]
  return args

def ext_modules():
  if not cuda_toolkit_available():
    return []

  return [
    CUDAExtension(
      "pytorch_custom_mma_cuda",
      sources = ["pytorch_custom_mma_cuda.cu"]
    )
  ]

# main setup code

setup(
  name = 'pytorch_custom_mma_cuda',
  packages = find_packages(exclude=[]),
  version = '0.0.3',
  license='MIT',
  install_requires=[
    'torch>=1.10',
    'torchtyping'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  ext_modules = ext_modules(),
  cmdclass = {"build_ext": BuildExtension},
  include_package_data = True,
)
