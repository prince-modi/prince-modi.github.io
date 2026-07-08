---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Hi, I'm Prince!
      username: admin

  - block: resume-skills
    id: skills
    title: Skills
    content:
      username: admin
    design:
      show_skill_percentage: false

  - block: resume-exp
    title: Education
    id: education
    content:
      items:
        - title: Master of Science in Computer Science
          icon_choice: "university"
          id: ucsd
          institution: University of California San Diego
          date_start: 2025-09-22
          date_end: 2027-06-01
          description: |
            Graduate student at UCSD, focusing on **Distributed Systems**
            
            **Relevant Courses:**
            - *Graduate Operating Systems*
            - *Computer Architecture*
            - *LLM Systems Optimization*

            **Projects:**
            - *System Measurement*: Developed a suite of micro-benchmarks to evaluate the Rockchip RK3588S SoC, utilizing ARMv8 cycle counters to measure CPU scheduling and OS primitive latencies with nanosecond precision.([Link](https://princemodi.me/post/system-measurement/))
 
        - title: Bachelor of Technology in Computer Engineering
          icon_choice: "university"
          id: guni
          institution: Ganpat University - UVPCE
          date_start: 2018-07-01
          date_end: 2022-06-01
          description: |
            **GPA:** *3.96/4.00*
            
            **Achievements:**
            - Academic scholarship for securing 2nd rank out of 60+ students
            
            **Relevant Courses:**
            - *Operating Systems*
            - *Computer Networks*
            - *Cloud Computing*
            - *Big Data Analytics*
            
            **Projects:**
            - *BitTorrent Client*: Built a peer-to-peer file-sharing client using Python's AsyncIO and BitTorrent protocol with a custom Bencode parser
            - *GIST*: Developed a YouTube video summarizer using NLTK, BART model, SQLite, and Tkinter

  - block: resume-exp
    title: Experience
    id: experience
    content:
      items:
        - title: 'Research Collaborator (Remote)'
          icon_choice: "university"
          id: iisc2
          company: Indian Institute of Science (IISc), Bangalore
          company_url: ''
          company_logo: ''
          date_start: 2024-04-30
          date_end: 2024-11-01
          location: Bangalore
          description: |
            - Authored system architecture and scalability analysis for Flotilla, securing acceptance into the Journal of Parallel and Distributed Computing (JPDC)
            - Orchestrated containerized deployments via custom Docker images across distributed clusters, validating system performance and scaling to 1,000+ concurrent clients
            
        - title: 'Research Associate (DREAM:Lab)'
          icon_choice: "university"
          id: iisc
          company: Indian Institute of Science (IISc), Bangalore
          company_url: ''
          company_logo: ''
          date_start: 2023-01-01
          date_end: 2024-04-30
          location: Bangalore
          description: |
            - Engineered Flotilla, an asynchronous federated learning framework in Python, optimizing distributed training across edge hardware topologies
            - Architected a custom Redis-based state store with periodic disk checkpointing, ensuring 100% state recovery and zero data loss during full server failures
            - Implemented gRPC and MQTT communication layers, minimizing coordination latency for federated learning message passing
            - Administered an 80-node heterogeneous edge cluster (Nvidia Jetson, Raspberry Pi), maintaining high availability for continuous systems research
            
        - title: Teaching Assistant, Data Engineering at Scale
          icon_choice: "university"
          company: Indian Institute of Science (IISc), Bangalore
          company_url: ''
          company_logo: ''
          date_start: 2023-08-01
          date_end: 2023-12-31
          location: Bangalore
          description: |
            - Instructed 40+ graduate students in distributed computing principles, covering HDFS, Map-Reduce, and Apache Spark architecture
            - Directed weekly lab sessions and restructured assignment parameters, increasing practical comprehension of large-scale data engineering
            
        - title: Software Engineering Intern (Intellza)
          id: intern
          company: Sterlite Technologies Limited, Ahmedabad
          company_url: ''
          company_logo: ''
          date_start: 2022-01-01
          date_end: 2022-08-31
          location: Ahmedabad
          description: |
            - Engineered core modules for Intellza, a unified data storage and analytics platform, utilizing NoSQL architectures
            - Integrated LiquiBase for continuous MongoDB schema tracking, eliminating version conflicts across deployment environments
            - Refactored CI/CD Docker image configurations, reducing image footprint by 65% and decreasing automated build times by 50%

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

  - block: collection
    id: posts
    content:
      title: Recent posts
      subtitle: ''
      text: ''
      count: 3
      filters:
        folders:
          - post
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      offset: 0
      order: desc
    design:
      view: card
      columns: '2'

  - block: resume-hobbies
    title: Hobbies
    id: hobbies
    content:
      items:
        - title: Reading
          icon_choice: "book-open-reader"
          description: "Currently reading: *A Wise Man's Fear* and *Thinking, Fast and Slow*"
        - title: Tennis & Pickleball
          icon_choice: "trophy"
          description: "Just enjoy hitting the ball around and having a good time with friends"
        - title: Formula 1
          icon_choice: "flag-checkered"
          description: "Following F1 races, team strategies, and technological advancements"
        - title: Home Lab
          icon_choice: "computer"
          description: "Running PiHole and PFSense, experimenting with network setups"
        - title: 3D Printing
          icon_choice: "print"
          description: "Designing and printing 3D models for personal projects or prototyping"
        - title: Astronomy
          icon_choice: "meteor"
          description: "Stargazing and exploring celestial bodies, learning about the universe"
---
