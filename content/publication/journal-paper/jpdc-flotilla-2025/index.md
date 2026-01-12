---
title: "Flotilla: A scalable, modular and resilient federated learning framework for heterogeneous resources"
authors:
- Roopkatha Banerjee
- admin
- Jinal Vyas
- Chunduru Sri Abhijit
- Tejus Chandrashekar
- Harsha Varun Marisetty
- Manik Gupta
- Yogesh Simmhan
date: "2025-05-14"
doi: "10.1016/j.jpdc.2025.105103"

publication_types: ["article-journal"]

publication: "*Journal of Parallel and Distributed Computing, Volume 203* [CORE A*]"
publication_short: "JPDC"

abstract: >
  We introduce *Flotilla*, a flexible and lightweight FL platform designed for real-world edge environments, offering modular strategy support, asynchronous updates, and high fault tolerance. It runs efficiently on edge hardware like Raspberry Pi and Jetson, outperforming or matching top frameworks like Flower, OpenFL, and FedML, while scaling seamlessly to 1000+ clients.

tags:
- Federated Learning
- Distributed Systems
- Applied Machine Learning

featured: true

# url_pdf: "https://authors.elsevier.com/c/1l8Ip_GwHPLUbc"
url_pdf: "https://arxiv.org/pdf/2507.02295"
url_code: "https://github.com/dream-lab/flotilla"

# Optional assets (uncomment if available)
# url_poster: "/uploads/hipcsrs-flotilla-poster.pdf"
# url_slides: "/uploads/hipcsrs_flotilla_presentation.pdf"

# image:
#   caption: "FL Framework Architecture"
#   focal_point: "center"
#   preview_only: false

# projects: ["project-name"]

# slides: "slides_filename"

show_related: true
---

  This builds further upon our initial model agnostic federated learning framework, discussed [here](https://princemodi.me/publication/conference-poster/hipc-srs-2023/). The framework described there was completely synchronous and did not support any checkpointing mechanism (yet). The framework we describe in this paper is almost a total rewrite. We moved away from a synchronous framework, deciding to add support for asynchronous federated learning strategies. We further implemented the newer version with a clearer separation of states in mind, which later helped us integrate checkpointing and reliability into this framework.

  ### My Contribution

  I was involved in the design and development of Flotilla from the ground-up. I implemented or was part of the group implementing 
  all the core components, and was also involved in the initial design and setup of the evaluation framework to study the system's
  behavior under expected, high-load, and failure conditions.

  Out of various components of this framework, one that I am most proud of is the Server Failure and Recovery using state checkpointing.
  We designed an external state store using Redis, that stores all the important state of the server during a machine learning training task.
  We also implemented a periodic disk-based checkpointing, as a back-up for our Redis state-store. Designing and testing this component was 
  the most fun I have had while working on this project. See Sections [3.5](https://arxiv.org/pdf/2507.02295#page=19&zoom=100,0,500) and [4.4](https://arxiv.org/pdf/2507.02295#page=31&zoom=100,0,400) in the paper for more details!

  I coauthored the initial drafts of the paper, focusing on articulating the systems perspective, including modular design,
  fault tolerance, and large-scale deployment challenges.
