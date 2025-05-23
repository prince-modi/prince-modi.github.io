---
title: "Flotilla: A scalable, modular and resilient federated learning framework for heterogeneous resources"
authors:
- Roopkatha Banerjee
- Prince Modi
- Jinal Vyas
- Chunduru Sri Abhijit
- Tejus Chandrashekar
- Harsha Varun Marisetty
- Manik Gupta
- Yogesh Simmhan
date: "2025-05-14"
doi: "https://doi.org/10.1016/j.jpdc.2025.105103"

publication_types: ["article-journal"]

publication: "*Journal of Parallel and Distributed Computing, Volume 203*"
publication_short: "JPDC"

abstract: >
  With the recent improvements in mobile and edge computing and rising concerns of data privacy, Federated Learning (FL) has rapidly gained popularity as a privacy-preserving, distributed machine learning methodology. Several FL frameworks have been built for testing novel FL strategies. However, most focus on validating the learning aspects of FL through pseudo-distributed simulation but not for deploying on real edge hardware in a distributed manner to meaningfully evaluate the federated aspects from a systems perspective. Current frameworks are also inherently not designed to support asynchronous aggregation, which is gaining popularity, and have limited resilience to client and server failures. We introduce Flotilla, a scalable and lightweight FL framework. It adopts a “user-first” modular design to help rapidly compose various synchronous and asynchronous FL strategies while being agnostic to the DNN architecture. It uses stateless clients and a server design that separates out the session state, which are periodically or incrementally checkpointed. We demonstrate the modularity of Flotilla by evaluating five different FL strategies for training five DNN models. We also evaluate the client and server-side fault tolerance on 200+ clients, and showcase its ability to rapidly failover within seconds. Finally, we show that Flotilla's resource usage on Raspberry Pis and Nvidia Jetson edge accelerators are comparable to or better than three state-of-the-art FL frameworks, Flower, OpenFL and FedML. It also scales significantly better compared to Flower for 1000+ clients. This positions Flotilla as a competitive candidate to build novel FL strategies on, compare them uniformly, rapidly deploy them, and perform systems research and optimizations.

tags:
- Federated Learning
- Distributed Systems
- Applied Machine Learning

featured: true

url_pdf: "https://doi.org/10.1016/j.jpdc.2025.105103"
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
