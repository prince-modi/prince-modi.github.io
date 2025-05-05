---
title: Exploring the Power of Google Spanner for Distributed Transaction Processing
summary: '*An overview of Google Spanner’s architecture, distributed transactions, and the TrueTime API, based on the paper by Corbett et al.*'
date: 2024-03-05
math: true
toc: true
image:
  placement: 2
  caption: 'Photo by Pawel Czerwinski on Unsplash'
tags:
- distributed systems
- databases
- Google Spanner
- research
---


This article is my understanding of distributed transactions and various aspects of **Spanner**, such as its structure, how it handles transactions like Read-Write and Read-Only Transactions, and the **TrueTime API**.

---

## Distributed Transactions

Transactions package multiple operations on data records to ensure they execute as a single unit, even in the event of failure. These guarantees are often referred to as **ACID** properties:

- **Atomic**: All-or-nothing
- **Consistent**: The system must remain in a valid state before and after
- **Isolated**: Each transaction appears to run alone
- **Durable**: Changes persist despite failures

---

## Need for Distributed Transactions

Large databases often contain millions of records and are **sharded** across machines (e.g., one shard contains rows A–N, another O–Z) to improve performance. When a transaction spans multiple shards, we need **concurrency control** and **commit protocols** to preserve ACID guarantees.

Spanner addresses this by using **Two-Phase Locking (2PL)** and **Two-Phase Commit (2PC)**.

---

## Spanner Architecture

A **Spanner deployment** is called a **universe**. It is composed of multiple **zones**, each with:

- A **zonemaster** that assigns data to spanservers
- Between 100 and several thousand **spanservers**

Each spanserver manages **100–1000 tablets**, which are sharded partitions of tables based on primary keys.

To support replication, Spanner:

- Implements a **Paxos state machine** per tablet
- Replicates tablets across spanservers using **Paxos groups**
- Handles concurrency control via a **lock table** at the Paxos leader
- Uses a **transaction manager** to coordinate multi-tablet transactions

---

## TrueTime API

Spanner relies on the **TrueTime API** to support external consistency and concurrency control.

- TrueTime returns an interval: `TTinterval = [earliest, latest]`
- `TT.now()` returns the interval during which the call occurred
- `TT.after(t)` returns true if `t` has definitely passed
- `TT.before(t)` returns true if `t` has definitely not yet occurred

TrueTime uses **GPS** and **atomic clocks**, each with different failure modes. Each data center has **time master machines**, and each machine runs a **timeslave daemon** that polls these masters.

> In production, the uncertainty bound ε (epsilon) is typically 1–7 ms, representing half the width of the TTinterval.

---

## Transactions in Spanner

### Read-Write Transactions

Spanner uses **Two-Phase Locking (2PL)** and **Two-Phase Commit (2PC)** for distributed read-write transactions.

- The coordinator gathers:
  - Prepared timestamps from non-coordinators
  - `TTcommit`, the commit time from the client
- It then chooses a commit timestamp that is:
  - Greater than all prepared timestamps
  - Greater than `TTcommit.latest`
  - Greater than any earlier transaction timestamps

Spanner performs a **commit wait** to ensure this timestamp is safely in the past before finalizing the commit.

### Read-Only Transactions

- If all required keys reside within a **single Paxos group**, the leader assigns the **last committed write timestamp** as the transaction timestamp, minimizing wait time.
- If keys span **multiple Paxos groups**, the timestamp is set to `TT.now().latest`, requiring a small delay until this time safely passes.

Read-only transactions can be served from **sufficiently up-to-date replicas**, allowing for **non-blocking and lock-free reads**.

---

## Summary

Spanner blends concepts from database and distributed systems research:

- From **databases**: SQL-like interface, relational schema, transactions
- From **systems**: Scalability, fault tolerance, sharding, replication, and global distribution

Thanks to the **TrueTime API**, Spanner achieves strong guarantees around **external consistency**, **lock-free read-only transactions**, and **non-blocking reads in the past**—demonstrating that precise time semantics are practical and powerful in distributed systems.

---
