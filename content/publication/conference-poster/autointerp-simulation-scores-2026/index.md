---
title: "Autointerp Simulation Scores Track Token Naming, Not Explanation Truth"
authors:
- Ayush Satish Govind
- admin
- Ishit Gulati
date: "2026-09-02"

publication_types: ["poster-conference"]
publication: "NeurIPS 2026 Workshop on InterpScience (submission)"
publication_short: "NeurIPS InterpScience 2026"

abstract: "Autointerp simulation scoring is used as a quality signal for sparse autoencoder latent descriptions, a use that presumes the score tracks whether a description is correct. We test that presumption with a within-latent 2x2 over whether a description names the latent's trigger token and whether its semantic gloss is true, on 16 token-driven latents of a Gemma-2-2B layer-12 GemmaScope SAE. Naming the token raises the score by 0.305, and does so for all 16 latents. Conditional on naming, swapping a true gloss for a false one changes it by 0.021 (95% CI -0.11 to +0.15). A judge-free baseline that marks tokens whose surface form appears in the description reproduces the effect. The results suggest that a high score can reflect naming the right token rather than truth of the description."

tags:
- Mechanistic Interpretability
- Sparse Autoencoders
- Evaluation

featured: true
url_pdf: "https://openreview.net/pdf?id=ZyKfnXzAcP"
show_related: true
---
