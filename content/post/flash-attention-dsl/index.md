---
title: "Modern GPU DSLs: Performance and Portability"
summary: 'Benchmarking RMSNorm and Flash Attention across Helion, Triton, Gluon, and CuTe on four NVIDIA GPU generations.'
date: 2026-03-17
math: true
toc: false
tags:
- GPU kernels
- LLM systems
- Triton
- CUDA
- performance analysis
---
For my CSE 291P final project at UCSD, our team benchmarked how GPU domain-specific languages (DSLs) trade off performance, productivity, and portability. We implemented RMSNorm (memory-bound) and Flash Attention-2 (compute-bound) in four DSLs, Helion, Triton, Gluon, and CuTe, and ran them across four GPU generations: Tesla T4, RTX 4060 Ti, A100, and RTX 5090.

## What We Measured

Each DSL sits at a different point on the abstraction spectrum, from Helion (PyTorch-like, auto-tuned, compiles down to Triton) to CuTe (near-CUDA control over memory layouts, thread mapping, and MMA/copy atoms). We measured implementation complexity (lines of code, cyclomatic complexity), raw throughput (GB/s for RMSNorm, TFLOPS for Flash Attention), and how consistently each kernel's hardware utilization held up when moved to a different GPU, using NVIDIA Nsight Compute profiles for compute and memory throughput.

## What We Found

Implementation complexity scaled roughly exponentially as abstraction decreased, CuTe kernels needed an order of magnitude more code than Torch or Helion. Performance mostly tracked that same trend on the A100, with CuTe leading on both kernels, but with notable exceptions: Helion's RMSNorm outperformed both Triton and Gluon by loading each row into SRAM once instead of twice, and CuTe's Flash Attention advantage only showed up at longer context lengths.

Portability told a different story. Triton and Gluon kernels tuned for the A100 lost significant memory throughput percentage when run on the T4, since their tile configurations were tuned for a GPU with more shared memory and stronger tensor cores. Helion, by contrast, held up more consistently across devices because its auto-tuner re-selects configurations per target rather than relying on one hand-tuned kernel, though at the cost of expensive tuning time.

---
*This was a course project for CSE 291P at UC San Diego, built with Hayden Prairie, Bhrugu Bharathi, Revant Mahajan, Cody Wang, Dario Wisznewer, and Rishabh Chittaranjan. I have the full report with additional figures and NCU analysis available on request, or check out the code at [github.com/prince-modi/gpu-kernel-dev](https://github.com/prince-modi/gpu-kernel-dev).*
