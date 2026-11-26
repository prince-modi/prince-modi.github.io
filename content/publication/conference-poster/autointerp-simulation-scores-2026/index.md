---
title: "Autointerp Simulation Scores Track Token Naming, Not Explanation Truth"
authors:
- Ayush Satish Govind
- admin
- Ishit Gulati
date: "2026-09-02"

publication_types: ["poster-conference"]
publication: "NeurIPS 2026 Workshop on Interpretability as a Science"
publication_short: "NeurIPS InterpScience 2026"
venue_tag: "NeurIPS-W"

abstract: "Autointerp simulation scoring is used as a quality signal for sparse autoencoder latent descriptions, a use that presumes the score tracks whether a description is correct. We test that presumption with a within-latent 2x2 over whether a description names the latent's trigger token and whether its semantic gloss is true, on 16 token-driven latents of a Gemma-2-2B layer-12 GemmaScope SAE. Naming the token raises the score by 0.305, and does so for all 16 latents. Conditional on naming, swapping a true gloss for a false one changes it by 0.021 (95% CI -0.11 to +0.15). A judge-free baseline that marks tokens whose surface form appears in the description reproduces the effect. The results suggest that a high score can reflect naming the right token rather than truth of the description."

tags:
- Mechanistic Interpretability
- Sparse Autoencoders
- Evaluation

featured: true
url_pdf: "/uploads/neurips_interpscience_2026.pdf"
show_related: true
---

Sparse autoencoders split a language model's activations into far more latents than anyone can read by hand, so the descriptions attached to them are written and checked automatically. The usual check is simulation scoring, where a judge model reads a description and predicts how strongly the latent fires on each token of its top-activating text. The correlation between those predictions and the real activations becomes the score. These scores are shown next to latents on Neuronpedia and used to compare SAEs, and both uses read a high score as evidence that the description is correct.

We wanted to see whether that holds. For 16 token-driven latents in a Gemma-2-2B layer-12 GemmaScope SAE, we wrote four descriptions per latent, varying two things independently: whether the description names the latent's trigger token, and whether its explanation is true. For example, a latent that fires on "Mary" scored 0.93 for "References to Mary-themed astronomical terminology and planetary observations", which is false, and -0.01 once "Mary" was removed from that same description.

Across all 16 latents, naming the token raised the score by 0.305. Swapping a true explanation for a false one, when the token was named, changed it by only 0.021. A simple baseline with no judge model, which just marks tokens that appear in the description, reproduced the naming effect and roughly matched the judge whenever the token was named. The judge does pick up on meaning when no token is named, so it isn't blind to semantics, but it leans on the token when one is there.

The scope is narrow: one model, one SAE layer, one judge, one prompt, and latents chosen because the token nearly is the explanation. Within that scope, a high simulation score tells you the description names the right token, not that the description is correct.
