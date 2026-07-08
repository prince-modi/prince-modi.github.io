---
# Display name
title: Prince Modi

# Name pronunciation (optional)
name_pronunciation: 

# Full name (for SEO)
first_name: Prince Bhagirathbhai
last_name: Modi

# Status emoji
status:
  icon: 

# Is this the primary user of the site?
superuser: true

# Role/position/tagline
role: Master's Student

# Organizations/Affiliations to show in About widget
organizations:
  - name: University of California San Diego
    url: https://cse.ucsd.edu/

# Short bio (displayed in user profile at end of posts)
bio: Distributed Systems@UCSD

# Skills
skills:
  - name: 'Proficient'
    items:
      - name: Python
        description: 'Used extensively in development of Flotilla and various personal projects'
        icon: 'fa-brands fa-python'
      - name: Docker
        description: 'Used heavily across jobs and personal projects'
        icon: 'fa-brands fa-docker'
      - name: Linux
        description: 'Daily use across work and personal setups (Arch BTW😝)'
        icon: 'fa-brands fa-linux'
      - name: NeoVim
        description: 'Daily driver, with custom plugins developed for specific workflows'
        icon: 'fa-solid fa-file-code'

  - name: 'Intermediate'
    items:
      - name: gRPC
        description: 'Integrated into a custom distributed framework using ProtoBuf definitions'
        icon: 'fa-solid fa-diagram-project'
      - name: MQTT
        description: 'Used while building a custom distributed framework and homelab projects'
        icon: 'fa-solid fa-diagram-project'
      - name: Redis
        description: 'Built a simplified version in Python to understand the internals'
        icon: 'fa-solid fa-database'
      - name: Git
        description: 'Implemented a basic version in Python to understand internal mechanisms'
        icon: 'fa-brands fa-git'

  - name: 'Familiar'
    items:
      - name: Go
        description: 'Explored through personal projects focused on distributed systems'
        icon: 'fa-brands fa-golang'
      - name: PyTorch
        description: 'Built CNNs, LSTMs, and custom dataloaders for a custom federated learning framework'
        icon: 'fa-solid fa-brain'
      - name: Lua
        description: 'Used to write custom NeoVim plugins and enhance editor behavior'
        icon: 'fa-solid fa-code'
      - name: MongoDB
        description: 'Used during an internship while working with NoSQL data models'
        icon: 'fa-solid fa-database'

social:
  - icon: envelope
    icon_pack: fas
    link: 'mailto:princebmodi@outlook.com'
  - icon: linkedin
    icon_pack: fab
    link: https://www.linkedin.com/in/modi-prince/
  - icon: graduation-cap 
    icon_pack: fas
    link: https://scholar.google.com/citations?user=nZAmHBcAAAAJ&hl=en
  - icon: github
    icon_pack: fab
    link: https://github.com/prince-modi
  - icon: cv
    icon_pack: ai
    link: uploads/CV.pdf
  - icon: file-lines
    icon_pack: fas
    link: uploads/resume.pdf

highlight_name: true
---

I'm a first-year Master's student in the Department of Computer Science and Engineering at UC San Diego, specializing in Distributed Systems. I have hands-on experience working on scalable and resilient systems, including development of [*Flotilla*](https://github.com/dream-lab/flotilla/), a modular federated learning framework, during my time at the Indian Institute of Science. While working on Flotilla, I became especially interested in the challenges of fault tolerance, consistency, and recovery in distributed systems. My broader interests span distributed computing, datacenter systems, and operating systems.

<div class="container">
  <div class="row">
    <!-- Experience Column -->
    <div class="col-12 col-md-4 mb-4">
      <h3>Experience</h3>
      <ul class="fa-ul">
        <li>
          <span class="fa-li"><i class="fas fa-briefcase"></i></span>
          <strong>Research Collaborator</strong><br/>
          <span style="font-size: 0.85em; color: #666;">IISc (2024)</span><br/>
          <span style="font-size: 0.9em;">
            - Authored system architecture and scalability analysis for Flotilla, securing acceptance into the <em>Journal of Parallel and Distributed Computing (JPDC)</em>.<br/>
            - Validated system performance by orchestrating containerized deployments across distributed clusters, successfully scaling to 1,000+ concurrent clients.
          </span>
        </li>
        <li style="margin-top: 15px;">
          <span class="fa-li"><i class="fas fa-briefcase"></i></span>
          <strong>Research Associate</strong><br/>
          <span style="font-size: 0.85em; color: #666;">IISc (2023–2024)</span><br/>
          <span style="font-size: 0.9em;">
            - Engineered Flotilla, an asynchronous federated learning framework, optimizing distributed training across edge hardware topologies.<br/>
            - Architected a custom Redis-based state store with disk checkpointing, achieving 100% state recovery and zero data loss during full server failures.<br/>
            - Administered an 80-node heterogeneous edge cluster (Nvidia Jetson, Raspberry Pi), maintaining high availability for continuous systems research.
          </span>
        </li>
        <li style="margin-top: 15px;">
          <span class="fa-li"><i class="fas fa-briefcase"></i></span>
          <strong>Software Eng. Intern</strong><br/>
          <span style="font-size: 0.85em; color: #666;">Sterlite Tech Ltd. (2022)</span><br/>
          <span style="font-size: 0.9em;">
            - Engineered core modules for Intellza, a unified data storage and analytics platform, utilizing NoSQL architectures.<br/>
            - Eliminated deployment version conflicts by integrating LiquiBase for continuous MongoDB schema tracking.<br/>
            - Refactored CI/CD Docker image configurations, reducing image footprint by 65% and decreasing automated build times by 50%.
          </span>
        </li>
      </ul>
    </div>

    <!-- Education Column -->
    <div class="col-12 col-md-4 mb-4">
      <h3>Education</h3>
      <ul class="fa-ul">
        <li>
          <span class="fa-li"><i class="fas fa-graduation-cap"></i></span>
          <strong>MSCS, UCSD</strong><br/>
          <span style="font-size: 0.85em; color: #666;">(2025–2027)</span>
        </li>
        <li style="margin-top: 10px;">
          <span class="fa-li"><i class="fas fa-graduation-cap"></i></span>
          <strong>B.Tech CE, UVPCE</strong><br/>
          <span style="font-size: 0.85em; color: #666;">(2018–2022)</span>
        </li>
      </ul>
    </div>

    <!-- Interest Column -->
    <div class="col-12 col-md-4 mb-4">
      <h3>Interests</h3>
      <ul class="fa-ul">
        <li class="mb-3">
          <span class="fa-li"><i class="fas fa-server"></i></span>
          Distributed Systems
        </li>
        <li class="mb-3">
          <span class="fa-li"><i class="fas fa-cloud"></i></span>
          Datacenter Systems
        </li>
        <li class="mb-3">
          <span class="fa-li"><i class="fas fa-computer"></i></span>
          Operating Systems
        </li>
      </ul>
    </div>
  </div>
</div>