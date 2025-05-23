---
title: "Towards a Modular Federated Learning Framework on Edge Devices"
authors:
- Roopkatha Banerjee
- admin
- Harsha Varun Marisetty
- Manik Gupta
- Yogesh Simmhan
date: "2023-12-18"

# Schedule page publish date (NOT publication's date).
# publishDate: "2025-05-04T23:24:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["poster-conference"]

# Conference name and optional abbreviated conference name.
publication: "30th IEEE International Conference on High Performance Computing, Data, and Analytics (2023)"
publication_short: "HiPC"

abstract: 'In this poster, we introduce *Flotilla*, a modular, model-agnostic Federated Learning (FL) framework that supports synchronous client-selection and aggregation strategies, and FL model deployment and training on edge client clusters, with telemetry for advanced systems research.'

tags:
- Federated Learning
- Distributed Systems
- Applied Machine Learning

featured: true

# Links (Optional)
url_pdf: '/uploads/hipcsrs_2023_fedml.pdf'
url_code: 'https://github.com/dream-lab/flotilla'
url_poster: '/uploads/hipcsrs-flotilla-poster.pdf'
url_slides: '/uploads/hipcsrs_flotilla_presentation.pdf'

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

### Framework Architecture
![Figure 1.](hipcsrs_2023_fedml.png)

*Flotilla* is designed to be modular and model-agnostic, enabling researchers to plug in different client selection and aggregation strategies for Federated Learning (FL). It supports edge-based client clusters like Raspberry Pis and integrates telemetry for advanced experimentation and systems-level insight. The framework is implemented in Python and provides abstraction layers for server orchestration, model deployment, and synchronous FL rounds.

### Experimental Insights

We evaluated different client selection strategies in a lab testbed comprising homogeneous Raspberry Pi devices connected via Gigabit LAN. In an IID data setting, where local models are inherently aligned, the **Default** strategy showed the fastest convergence due to receiving the highest number of updates per round.

**Random Selection (RS)** and **Partial Heterogeneous Loss (PHL)** performed similarly, as client validation losses were nearly identical across rounds. The absence of stragglers ensured that training durations remained consistent across clients. However, PHL showed periodic increases in training time due to validation overhead every alternate round.

The model converged in **100 FL rounds**, reaching **98.4% accuracy**, with an **average round time of 375 seconds**.

![Figure 2.](hipcsrs_2023_fedml_training_graph.png)

### Press
* ["Towards a Modular Federated Learning Framework on Edge Devices"](https://2023.hipc.org/srs-2023/#:~:text=Towards%20a%20Modular%20Federated%20Learning%20Framework%20on%20Edge%20Devices) at HiPC 2023 Student Research Symposium
