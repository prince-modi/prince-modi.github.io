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
doi: "https://doi.org/10.1016/j.jpdc.2025.105103"

# Schedule page publish date (NOT publication's date).
# publishDate: "2025-05-04T23:24:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["article-journal"]

# Publication name and optional abbreviated publication name.
publication: "*Journal of Parallel and Distributed Computing, Volume*(203)"
publication_short: "JPDC"


abstract: 'With the recent improvements in mobile and edge computing and rising concerns of data privacy, Federated Learning (FL) has rapidly gained popularity as a privacy-preserving, distributed machine learning methodology. Several FL frameworks have been built for testing novel FL strategies. However, most focus on validating the learning aspects of FL through pseudo-distributed simulation but not for deploying on real edge hardware in a distributed manner to meaningfully evaluate the federated aspects from a systems perspective. Current frameworks are also inherently not designed to support asynchronous aggregation, which is gaining popularity, and have limited resilience to client and server failures. We introduce Flotilla, a scalable and lightweight FL framework. It adopts a “user-first” modular design to help rapidly compose various synchronous and asynchronous FL strategies while being agnostic to the DNN architecture. It uses stateless clients and a server design that separates out the session state, which are periodically or incrementally checkpointed. We demonstrate the modularity of Flotilla by evaluating five different FL strategies for training five DNN models. We also evaluate the client and server-side fault tolerance on 200+ clients, and showcase its ability to rapidly failover within seconds. Finally, we show that Flotilla's resource usage on Raspberry Pis and Nvidia Jetson edge accelerators are comparable to or better than three state-of-the-art FL frameworks, Flower, OpenFL and FedML. It also scales significantly better compared to Flower for 1000+ clients. This positions Flotilla as a competitive candidate to build novel FL strategies on, compare them uniformly, rapidly deploy them, and perform systems research and optimizations.'

tags:
- Federated Learning
- Distributed Systems
- Applied Machine Learning

featured: true

# Links (Optional)
url_pdf: 'https://doi.org/10.1016/j.jpdc.2025.105103'
url_code: 'https://github.com/dream-lab/flotilla'
# url_poster: '/uploads/hipcsrs-flotilla-poster.pdf'
# url_slides: '/uploads/hipcsrs_flotilla_presentation.pdf'

# Featured image
# image:
#   caption: 'FL Framework Architecture'
#   focal_point: "center"
#   preview_only: false

# Associated Projects (optional).
# projects: ['project-name']

# Slides (optional).
# slides: "slides_filename"

# Other options
show_related: true

---

# ### Framework Architecture
# ![Figure 1.](hipcsrs_2023_fedml.png)

#### Highlights

• Flotilla enables modular and intuitive composition of federated learning strategies.
• It supports both synchronous and asynchronous strategies on diverse edge devices.
• It handles both client and server failures for resilient training.
• It scales efficiently to 1000+ clients and exhibits superior weak-scaling.

# ### Experimental Insights

# We evaluated different client selection strategies in a lab testbed comprising homogeneous Raspberry Pi devices connected via Gigabit LAN. In an IID data setting, where local models are inherently aligned, the **Default** strategy showed the fastest convergence due to receiving the highest number of updates per round.

# **Random Selection (RS)** and **Partial Heterogeneous Loss (PHL)** performed similarly, as client validation losses were nearly identical across rounds. The absence of stragglers ensured that training durations remained consistent across clients. However, PHL showed periodic increases in training time due to validation overhead every alternate round.

# The model converged in **100 FL rounds**, reaching **98.4% accuracy**, with an **average round time of 375 seconds**.

# ![Figure 2.](hipcsrs_2023_fedml_training_graph.png)

# ### Press
# * ["Towards a Modular Federated Learning Framework on Edge Devices"](https://2023.hipc.org/srs-2023/#:~:text=Towards%20a%20Modular%20Federated%20Learning%20Framework%20on%20Edge%20Devices) at HiPC 2023 Student Research Symposium
