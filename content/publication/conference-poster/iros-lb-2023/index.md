---
title: "Towards Collision Avoidance for UAVs to Guide the Visually Impaired"
authors:
- Suman Raj
- Swapnil Padhi
- Ruchi Bhoot
- admin
- Yogesh Simmhan
date: "2023-08-10"

# Schedule page publish date (NOT publication's date).
# publishDate: "2025-05-04T23:24:00Z"

# Publication type.
# publication_types: ["poster-conference"]

# Conference name and optional abbreviated conference name.
conference: "IEEE/RSJ International Conference on Intelligent Robots and Systems (2023)"
conference_short: "IROS"

abstract: "In this poster, we explore a mechanism to detect obstacles within a distance 'd' ahead of a visually impaired person (VIP) and offer them a path with a minimum width 'w' to navigate between the obstacles."

tags:
- Drones
- Distributed Systems
- Applied Machine Learning
- CNNs

featured: true

# Links (Optional)
# url_pdf: '/uploads/iros_2023_uav_vip.pdf'
# url_code: 'https://github.com/dream-lab/uav-vip-collision-avoidance'
# url_poster: '/uploads/iros-poster.pdf'
# url_slides: '/uploads/iros-presentation.pdf'

# Other options
show_related: true

---

### System Overview

We present an assistive navigation system using **Unmanned Aerial Vehicles (UAVs)** to support **visually impaired persons (VIPs)** in obstacle-rich environments. The UAV **flies behind the VIP**, passively tracking their movement while scanning the path ahead for obstacles within a predefined distance `d`.

Using onboard sensors, the UAV detects objects and identifies open pathways with a minimum required width `w`. This enables it to guide the VIP with timely feedback, helping them avoid collisions and stay within safe navigable zones.

### Methodology

The onboard edge hardware, carried by the VIP, handles all processing to maintain low latency. The processing stack includes lightweight **Convolutional Neural Networks (CNNs)** to identify and localize objects in the field of view, along with **depth perception** to calculate the distance of these objects from the VIP.

Using this data, a path planning module computes feasible routes by analyzing gaps between obstacles and determining navigable regions.

Guidance cues are delivered to the VIP through **audio** or **haptic feedback systems**, offering intuitive, non-visual direction changes during navigation.

### Outcomes

Initial experiments in controlled settings show that the system successfully maintains a safe buffer around obstacles and consistently identifies viable paths. It adapts effectively to cluttered environments while remaining compact and responsive.

This approach demonstrates the viability of **UAV-based assistive mobility** systems that are both autonomous and user-centric, offering a promising solution for real-world guidance of visually impaired individuals. Future work includes outdoor testing, dynamic obstacle handling, and longitudinal usability studies with real users.
