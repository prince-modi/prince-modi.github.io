---
title: ''
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Hi, I'm Prince!
      username: admin

  - block: resume-exp
    title: Experience
    id: experience
    content:
      items:
        - title: 'Research Collaborator (Remote)'
          company: Indian Institute of Science (IISc), Bangalore
          date_start: 2024-04-30
          date_end: 2024-11-01
          description: |
            - Authored system architecture and scalability analysis for Flotilla, securing acceptance into the *Journal of Parallel and Distributed Computing (JPDC)*.
            - Validated system performance by orchestrating containerized deployments across distributed clusters, scaling to 1,000+ concurrent clients.
            
        - title: 'Research Associate (DREAM:Lab)'
          company: Indian Institute of Science (IISc), Bangalore
          date_start: 2023-01-01
          date_end: 2024-04-30
          description: |
            - Engineered Flotilla, an asynchronous federated learning framework in Python, optimizing distributed training across edge hardware topologies.
            - Architected a Redis-based state store with disk checkpointing, ensuring 100% state recovery and zero data loss during server failures.
            - Administered an 80-node heterogeneous edge cluster (Nvidia Jetson, Raspberry Pi), maintaining high availability for continuous systems research.

  - block: resume-exp
    title: Education
    id: education
    content:
      items:
        - title: Master of Science in Computer Science
          institution: University of California San Diego
          date_start: 2025-09-22
          date_end: 2027-06-01
          description: Focusing on **Distributed Systems** and **LLM System Optimization**.
          
        - title: Bachelor of Technology in Computer Engineering
          institution: Ganpat University - UVPCE
          date_start: 2018-07-01
          date_end: 2022-06-01
          description: GPA 3.96/4.00.

  - block: collection
    id: featured-publications
    content:
      title: Featured publications
      filters:
        folders:
          - publication
        featured_only: true
    design:
      columns: '2'
      view: card
---