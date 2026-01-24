---
title: "Peeling Back the Layers: An Empirical Analysis of Modern ARM Primitives"
summary: 'A performance characterization of the Rockchip RK3588S SoC, exploring the gap between theoretical hardware limits and observed system behavior.'
date: 2026-01-23
math: true
toc: true
tags:
- systems research
- benchmarking
- ARM architecture
- operating systems
- performance analysis
---

There's a gap between what hardware can do and what software actually achieves. For my CSE 221 System Measurement project at UCSD, I used an Orange Pi 5 (Rockchip RK3588S) to characterize the performance of a modern ARM-based single-board computer.

The project required building micro-benchmarks from scratch to measure CPU scheduling, OS services, memory hierarchy, networking, and file system performance.

> **Note:** Per course policy, I'm not sharing implementation details, methodologies, or results. If you want to discuss approaches feel free to reach out directly.

## What I Measured

The Rockchip RK3588S uses Arm's Big.LITTLE architecture, combining Cortex-A76 performance cores with Cortex-A55 efficiency cores. Running Armbian Linux, this system provided an interesting platform for understanding modern ARM performance characteristics.

The project covered several categories:

**CPU and OS Services**
- Measurement overhead and loop costs
- Procedure call overhead
- System call overhead  
- Process and thread creation time
- Context switching costs

**Networking**
- Round trip time (application-level and ICMP)
- Peak bandwidth
- Connection setup and teardown overhead
- TCP vs. loopback performance

## The Process

Each measurement required:
1. Estimating base hardware performance from specifications
2. Predicting software overhead
3. Implementing the benchmark
4. Analyzing the results

The challenge wasn't writing the benchmarks. Understanding what you're actually measuring at nanosecond timescales requires careful attention to hardware behavior, compiler optimizations, and OS scheduling effects.

## What I Learned

This project changed how I think about system performance. Theoretical models are useful starting points, but modern hardware behavior is complex. Features like superscalar execution, out-of-order processing, prefetching, and multi-level caching create a large gap between simple predictions and reality.

Benchmarking itself is a skill. The measurement process can easily introduce artifacts that swamp the signal you're trying to measure. Getting reliable, repeatable results requires understanding your tools and the system you're measuring.

The exercise also highlighted the costs of abstraction. Every OS primitive has overhead. Every system call crosses a protection boundary. Every context switch involves state management. These costs add up quickly in real applications.

## Reflections

Working on bare metal measurements gave me a deeper appreciation for:
- The engineering tradeoffs in hardware design
- Why certain software design patterns exist (thread pools, buffer caching, etc.)
- The complexity hidden behind simple operations
- The importance of empirical measurement over assumptions

---

*This work was completed as part of CSE 221 (Operating Systems) at UC San Diego. I have a detailed technical report documenting the full methodology, results, and analysis. If you'd like to discuss this project or see the report, please contact me directly.*
