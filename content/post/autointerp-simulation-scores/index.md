---
title: "Autointerp Simulation Scores Track Token Naming, Not Explanation Truth"
summary: 'Our NeurIPS 2026 InterpScience workshop paper on what autointerp simulation scores actually respond to.'
date: 2026-11-25
math: false
toc: false
tags:
- Mechanistic Interpretability
- Sparse Autoencoders
- Evaluation
---

Our paper with Ayush Satish Govind and Ishit Gulati, "Autointerp Simulation Scores Track Token Naming, Not Explanation Truth", is in the [NeurIPS 2026 Workshop on Interpretability as a Science](/uploads/neurips_interpscience_2026.pdf).

Sparse autoencoders split a language model's activations into far more latents than anyone can read by hand, so the descriptions attached to them are written and checked automatically. The usual check is simulation scoring, where a judge model reads a description and predicts how strongly the latent fires on each token of its top-activating text. The correlation between those predictions and the real activations becomes the score. These scores are shown next to latents on Neuronpedia and used to compare SAEs, and both uses read a high score as evidence that the description is correct.

We wanted to see whether that holds. For 16 token-driven latents in a Gemma-2-2B layer-12 GemmaScope SAE, we wrote four descriptions per latent, varying two things independently: whether the description names the latent's trigger token, and whether its explanation is true. For example, a latent that fires on "Mary" scored 0.93 for "References to Mary-themed astronomical terminology and planetary observations", which is false, and -0.01 once "Mary" was removed from that same description.

Across all 16 latents, naming the token raised the score by 0.305. Swapping a true explanation for a false one, when the token was named, changed it by only 0.021. A simple baseline with no judge model, which just marks tokens that appear in the description, reproduced the naming effect and roughly matched the judge whenever the token was named. The judge does pick up on meaning when no token is named, so it isn't blind to semantics, but it leans on the token when one is there.

The scope is narrow: one model, one SAE layer, one judge, one prompt, and latents chosen because the token nearly is the explanation. Within that scope, a high simulation score tells you the description names the right token, not that the description is correct.

The full paper is [here](/uploads/neurips_interpscience_2026.pdf).
