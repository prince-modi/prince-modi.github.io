---
title: "Dead Nodes Tell No Tales: Globally Consistent Snapshots for Distributed Containers"
summary: 'A reliable-UDP proxy and Chandy-Lamport based snapshotting system for checkpointing and restoring live, distributed containerized applications.'
date: 2026-06-09
math: false
toc: true
tags:
- distributed systems
- containers
- networking
- systems research
- fault tolerance
---
For my CSE 223B (Distributed Systems) project at UCSD, my team and I built a system for taking globally consistent snapshots of a distributed, containerized application while it continues running. The repo is [`dead-nodes-tell-no-tales`](https://github.com/nathanparekh/dead-nodes-tell-no-tales).

## Problem

Checkpointing a single container is a solved problem: CRIU and Podman handle capturing execution context and memory layout. Checkpointing several containers that are actively exchanging network traffic is harder, since packets in flight at the moment of checkpointing can end up on one side of the cut but not the other, producing checkpoints that don't represent any single valid state of the system as a whole.

We built the snapshot protocol on the Chandy-Lamport algorithm, which assumes in-order, lossless communication channels between nodes. UDP provides neither, so the project had two parts: a network layer that gives UDP the delivery guarantees Chandy-Lamport requires, and a mechanism for triggering privileged CRIU/Podman checkpoints on a container from an unprivileged sidecar.

## What We Built

**A reliable-UDP sidecar proxy.** Each application container runs alongside a sidecar that transparently intercepts its UDP traffic using `iptables`/`TPROXY` rules. Outgoing packets get sequence numbers and are buffered and retransmitted until acknowledged; incoming packets are acknowledged, reordered if needed, and delivered in sequence. Neither side times out, since Chandy-Lamport requires that no message is ever lost or reordered.

**Chandy-Lamport snapshotting, built into the same proxy.** On seeing a marker packet, a sidecar drains its buffers, triggers a CRIU checkpoint of its application container, and propagates the marker to its peers. It then collects in-flight messages on each incoming channel until every peer has confirmed, at which point the snapshot for that node is complete.

**Breakout Receiver**, a small REST service on the host that exposes checkpoint, restore, and container-management operations. This let unprivileged sidecars trigger privileged CRIU/Podman operations without needing root access themselves, after we ruled out mounting the Podman socket directly and per-container Unix pipes as less clean alternatives.

## Evaluation

We validated the system with a small distributed "Counter" application, where nodes transfer value between each other over UDP under a global invariant that should hold across any snapshot/restore cycle. With our sidecars running, the invariant held even under concurrent load and mid-transfer snapshots across a multi-host, multi-container deployment on AWS. Removing the sidecars and relying on plain CRIU checkpointing reliably broke the invariant, confirming that the reliable-UDP layer was doing real work rather than just adding overhead. The system also stayed correct under artificial network delay and heavy packet loss during testing.

---
*This was a course project for CSE 223B (Distributed Systems) at UC San Diego, built with Nathan Parekh and Reese Whitlock. I have a detailed technical report documenting the full design, evaluation, and results. If you'd like to discuss this project or see the paper, please reach out directly, or check out the code at [github.com/nathanparekh/dead-nodes-tell-no-tales](https://github.com/nathanparekh/dead-nodes-tell-no-tales).*
