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
            - Continued voluntary collaboration with the DREAM:Lab to finalize research validation for the JPDC 2025 publication
            - Authored the in-depth system architecture and scalability analysis for Flotilla, leading to its acceptance in the Journal of Parallel and Distributed Computing (JPDC)
            - Orchestrated containerized deployments using custom Docker images across distributed clusters to validate system performance on 1000+ concurrent clients
            
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
            **Responsibilities include:**
            - Built an asynchronous federated learning framework (Flotilla) in Python, optimized for edge hardware deployment
            - Implemented server and client sides using MQTT and gRPC for efficient message passing and coordination in federated learning
            - Designed a custom Redis-based state store with checkpointing to enable recovery from full server failures without data loss or disruption
            - Integrated client selection and aggregation strategies from current research for performance, accuracy, and turnaround optimization
            - Collaborated with PhD students under Prof. Manik Gupta (BITS Pilani) and Prof. Yogesh Simmhan (IISc) to ensure Flotilla's scalability and reliability
            - Configured and managed an 80+ node edge cluster (Nvidia Jetsons, Raspberry Pis), supporting lab infrastructure and projects including Flotilla
            
            **Volunteering:**
            - *Senior Student Volunteer*, Indian Institute of Science – IEEE/ACM CCGrid 2023: Co-organized a 300+ participant conference; coordinated 3 poster sessions and assisted keynote speakers and faculty
            - *Student Volunteer*, IISc Open Day 2023: Coordinated presentation sessions for DREAM:Lab projects
            
        - title: Teaching Assistant, Data Engineering at Scale
          icon_choice: "university"
          company: Indian Institute of Science (IISc), Bangalore
          company_url: ''
          company_logo: ''
          date_start: 2023-08-01
          date_end: 2023-12-31
          location: Bangalore
          description: |
            **Responsibilities include:**
            - Taught a graduate-level course to a class of 40+ students, comprising topics such as HDFS, Map-Reduce, Apache Spark
            - Facilitated and led a 2-hour lab session per week, prepared and graded assignments, conducted one-on-one office hours, and conducted doubt-clearing sessions
            
        - title: Software Engineering Intern (Intellza)
          id: intern
          company: Sterlite Technologies Limited, Ahmedabad
          company_url: ''
          company_logo: ''
          date_start: 2022-01-01
          date_end: 2022-08-31
          location: Ahmedabad
          description: |
            **Responsibilities include:**
            - Developed Intellza, a unified data storage and analytics platform, alongside a cross-functional team
            - Developed and integrated a module to maintain and track schema changes for MongoDB on Intellza using LiquiBase
            - Created Docker images and optimized the existing images as per Docker's recommendations, reducing the image size to 35% and improving the build times of the project's CI/CD pipeline by 50%

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