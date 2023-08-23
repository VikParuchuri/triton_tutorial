# Triton tutorials

[Triton](https://www.github.com/openai/triton) is a language for writing GPU kernels.  It's easier to use than CUDA, and interoperates well with PyTorch.

If you want to speed up PyTorch training or inference speed, you can try writing kernels for the heavier operations using Triton. ([flash attention](https://github.com/Dao-AILab/flash-attention) is a good example of a custom GPU kernel that speeds up training)

This repo has my notes as I learn to use Triton.  They include a lot of code, and some discussion of the key concepts.  They're geared towards people new to GPU programming and Triton.

Hopefully you will find them useful.

# Contents

1. [GPU Basics](01_gpu_basics.ipynb)
2. [Vector Addition](02_vector_addition.ipynb)
3. [Matrix Multiplication](03_small_matrix_multiplication.ipynb)
4. [Softmax forward and backward](04_softmax_fwd_bwd.ipynb)
5. [Block matmul](05_block_matmul.ipynb)
6. [Matmul forward and backward](06_matmul_fwd_bwd.ipynb)

# Install

To install Triton, just do `pip install triton`.  You need a CUDA-compatible GPU with CUDA installed to use it.

# References

Material in these notebooks came from the following sources (and they're generally good documentation):

- [Triton tutorials](https://triton-lang.org/main/index.html)
- [NVIDIA CUDA guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA MMM](https://siboehm.com/articles/22/CUDA-MMM)



