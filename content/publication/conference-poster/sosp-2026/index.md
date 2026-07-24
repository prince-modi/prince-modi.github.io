---
title: "Recovering Internal Command Interfaces from Applications for Direct Agent Invocation"
authors:
- admin
- Varad Kottawar
- Ayush Govind
date: "2026-07-23"

# Schedule page publish date (NOT publication's date).
# publishDate: "2025-05-04T23:24:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["poster-conference"]

# Conference name and optional abbreviated conference name.
publication: "32nd ACM Symposium on Operating Systems Principles (2026)"
publication_short: "SOSP"

abstract: "Computer-use agents built on LLMs mostly interact with desktop apps the way a person would: they look at a screenshot, figure out where a button is, and click it. Do that a hundred times and you can automate a workflow. But it's slow, brittle, and expensive. Every click is a round trip to the model. Every misread pixel is a chance for the whole chain to fall apart."

tags:
- Operating Systems
- Machine Learning
- Large Language Models

featured: true

# Links (Optional)
url_pdf: '/uploads/sosp_2026.pdf'

show_related: true

---


## The problem
 
Computer-use agents built on LLMs mostly interact with desktop apps the way a person would: they look at a screenshot, figure out where a button is, and click it. Do that a hundred times and you can automate a workflow. But it's slow, brittle, and expensive. Every click is a round trip to the model. Every misread pixel is a chance for the whole chain to fall apart.
 
Some recent work tries to shortcut this by using OS accessibility APIs instead of raw screen coordinates, letting the model say "set this checkbox" instead of "click at (412, 88)." That helps, but it only works as well as the app's accessibility metadata, which is often incomplete, and it still routes every action through a layer that sits outside the app's own logic.
 
That got us asking a more basic question: if the point is to let an LLM trigger application functionality directly, why go through the accessibility layer at all? Under the hood, most GUI controls and accessibility elements are just wrappers around ordinary internal function calls. What if we could recover and call those functions directly?
 
## The idea
 
We use [Frida](https://frida.re), a dynamic instrumentation toolkit, to hook into a running application, trace what happens when a person performs an action, and recover the actual internal functions responsible, bypassing the GUI and the accessibility tree entirely.
 
Concretely: attach Frida to a running app, use its Stalker component to record the execution trace while a human performs some action, then map the executed code back to symbols to find candidate functions. Once we've identified the right function and understand its calling convention, we can invoke it directly, no clicking required.

## Proof of concept: GIMP
 
We tested this on GIMP, hooking cropping, flipping, and rotating operations. A couple of things stood out:
 
- One operation (flip) turned out to be reached via a tail call rather than a normal call instruction, so it was invisible to our first tracing setup. Switching to block-level event tracing fixed this, but it was a good reminder that "just trace the calls" isn't always as simple as it sounds.
- We confirmed argument ordering empirically, by hooking a function and comparing the values we captured against known, operator-specified inputs (e.g. a crop at a known position and size).
- Once hooked, invoking the functions directly produced the expected visual result, and undo/redo behavior stayed correctly intact even when interleaved with normal GUI edits. That turned out to be because GIMP manages its own undo bracketing internally, so we got that behavior for free rather than having to reconstruct it ourselves.

## Does it actually help?
 
We gave a computer-use agent (built on `trycua/cua`, running Claude Code) access to our recovered hooks as an additional tool, alongside its normal GUI-based capabilities, and compared it against a GUI-only baseline on a simple crop-and-save task.
 
With the hook available, the agent used roughly a third of the tokens and completed the task about 2.7x faster than the GUI-only baseline, averaged over 10 runs.
 
That's a small result on a single app and a handful of operations, but it's a real signal that skipping the GUI/accessibility layers entirely, when possible, is worth pursuing further.
 
## Open questions
 
The obvious limitation: we did this by hand, for one application. A useful version of this idea needs some way to *automatically* discover internal command interfaces rather than a person manually tracing and hooking each function. We see two directions worth exploring:
 
- **Source available:** analyze the codebase directly to find candidate functions tied to user-facing actions.
- **Source unavailable:** explore the compiled binary automatically to find the same thing, which is a harder problem but avoids depending on source access.
There's also the open question of how well this generalizes. GIMP happened to manage its own undo state internally, which made our lives much easier. Not every application will be this well-behaved, and we don't yet know how much manual work would be needed to handle the ones that aren't.
 
## What's next
 
This started as a comment from Professor Alex Snoeren during a SysNet seminar talk, and turned into a poster that's now been accepted to [SOSP 2026](https://sigops.org/s/conferences/sosp/2026/) in Prague. It's early-stage work, not a finished system, but it's been a fun rabbit hole to go down, and we're looking forward to hearing what people think in September.
