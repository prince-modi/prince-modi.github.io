---
title: "Understanding VMware FT: Deterministic Replay and Continuous Availability in Virtualized Environments"
summary: '*An in-depth look at VMware Fault Tolerance (FT) and how it achieves transparent failover for virtual machines using deterministic replay, based on the paper by Scales, Nelson, and Venkitachalam.*'
date: 2022-09-01
math: false
toc: true
image:
  placement: 2
  caption: 'Image credit: vmware.com'
tags:
- fault tolerance
- virtual machines
- VMware FT
- deterministic replay
- distributed systems
- cloud computing
---

## Introduction

The paper [*The Design of a Practical System for Fault-Tolerant Virtual Machines*](http://nil.lcs.mit.edu/6.824/2020/papers/vm-ft.pdf) by Daniel J. Scales, Mike Nelson, and Ganesh Venkitachalam (VMware Inc.) presents an enterprise-grade system to provide fault-tolerant virtual machines (VMs) with continuous availability. The authors leverage a **primary/backup approach** in which the execution of a primary VM is **replicated on another VM** running on a separate server. If the primary fails, the backup can immediately take over—ensuring zero downtime and no data loss.

---

## Approaches to State Synchronization

For the primary/backup model to work, both servers must maintain an **identical state**. Two approaches to achieve this:

### 1. State Transfer

This method involves sending the entire state (CPU, memory, I/O) of the primary VM to the backup continuously.

- **Pros**: Simple to conceptualize.
- **Cons**: Requires **high bandwidth**, often impractical in real-world settings.

### 2. Replicated State Machine

This method uses deterministic state machines that receive identical inputs in the same order to remain synchronized.

- **Pros**: Requires **less bandwidth**.
- **Cons**: Real-world VMs are not fully deterministic, requiring mechanisms to **capture and replay non-determinism**.

---

## Why Use Virtualization?

Non-determinism due to varying processor frequencies or architectures makes deterministic replay challenging on physical machines. Using **virtual machines** running on a **hypervisor** simplifies synchronization and replay by abstracting away hardware differences.

---

## System Architecture

VMware FT is implemented on the **VMware vSphere 4.0** platform. Key architectural elements include:

- Fully virtualized **x86 VMs**.
- Shared virtual disks accessible by both primary and backup.
- Only the **primary VM** is visible to the network.
- All **input events** are logged and sent to the backup through a **logging channel**.
- The focus is on **network and disk traffic** as the main sources of input.

---

## Deterministic Replay

To maintain exact VM states, VMware FT:

- Captures all **non-deterministic inputs/events** (e.g., interrupts, timestamps).
- **Replays them deterministically** on the backup VM.
- Uses **hardware performance counters** (from AMD and Intel) to implement efficient replay.

---

## The FT Protocol

Consider a database update operation from 'foo' to 'bar'. If the client receives confirmation before the backup logs the event—and the primary fails—a state inconsistency occurs.

### Output Rule

To prevent this, VMware FT enforces the **Output Rule**:

> The primary VM must not send output to the external world until the backup VM has received and acknowledged the log entry associated with the operation.

This ensures **client-visible outputs are always replayable**.

---

## Failure Detection and Split-Brain Handling

VMs detect failure via **heartbeat and logging channel timeouts** (a few seconds).

### Split-Brain Problem

Occurs if the backup assumes failure due to a **network partition**, while the primary is still operational.

#### Solution:

- Uses **shared storage** as a **tiebreaker**.
- Performs an **atomic test-and-set** on shared storage to determine which VM may go live.
- The other VM **commits suicide** if it loses the race.

---

## Restoring Redundancy

After a failure:

- A **new backup VM** is started on a different host.
- This is achieved using **FT VMotion**, which:
  - Clones the current primary VM.
  - Sets up a new **logging channel**.
  - Puts the new VM in **replay mode**.

---

## Logging Channel and Execution Lag

Hypervisors use **large log buffers** to record and stream inputs:

- Execution lag between primary and backup is typically under **100 ms**.
- If lag grows, the **primary’s CPU is throttled** slightly.
- A **feedback loop** adjusts CPU usage to maintain synchronization.

---

## Design Alternatives: Shared vs. Non-Shared Disks

### Shared Disks

- Both VMs access the **same virtual disk**.
- **Pros**:
  - Naturally consistent.
  - Aids in split-brain resolution.
- **Cons**:
  - Writes must obey the **output rule**.

### Non-Shared Disks

- Each VM has its own copy of the disk.
- **Pros**:
  - Disk writes are internal state; no need for output rule.
- **Cons**:
  - Requires **explicit syncing** post-failure.
  - No shared medium to resolve split-brain; needs an external tiebreaker.

---

## Key Takeaways

- VMware FT uses **deterministic replay** for efficient fault-tolerant VM replication.
- The system provides **transparent failover** and **automatic redundancy restoration**.
- The **Output Rule** ensures consistency during failover.
- **Shared storage** plays a vital role in maintaining consistency and resolving split-brain.
- At the time of the paper, **only uni-processor VMs** were supported efficiently under deterministic replay.

---