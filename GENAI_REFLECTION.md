# Reflection on GenAI usage — Phase I

**Team:** AI Gangsta · **Author:** Anton Sadeq · **Status:** single active member; the "human-only
pair" of the assignment does not exist in this team (see `Group_situation`). This reflection therefore
cannot compare two pairs. Where the course asks for that comparison, I compare instead against how I
would have worked alone, which I can judge from the first days of the project before the assistant
was involved in the analysis.

The detailed prompt-by-prompt record is in `AI_INTERACTION_RECORD.md`; this file is the reflection on it.

## 1. How GenAI was used, by the five modes of the AI4VIS session

**#1 GenAI as a tool to transform data.** Used only to *build* data, never to *label* it. The
assistant wrote the inference scripts that turn images into masks, probabilities and per-image
tables, and the perturbation and object-level scripts. No AI-derived attribute (no LLM judgement, no
caption, no generated label) enters any figure, so the "GenAI-derived data is not ground truth"
warning of the course did not apply. The verification effort went into the scripts instead: they had
to reproduce the archived test score before anything was plotted, and they did (within 2e-5).

**#2 GenAI as a collaborator to design and implement visualisations.** This was the main use.
The pattern that worked: I described the question and the data columns, the assistant proposed a
chart form and an encoding, I accepted or redirected, it implemented, and then it looked at its own
rendered figure. Ideas that came from the assistant: the strip plot of per-image mIoU against the
pixel-weighted dataset score, the IoU-versus-log-pixel-count scatter, the split of every error type
into boundary and interior pixels, the reliability bars with a cumulative "errors caught" curve, the
detection-rate-by-object-size view and the greenness control. Ideas that came from me: the notebook as
the medium, the crop-green / weed-red convention, one frame per example instead of one frame per
class, and the decision to keep the analysis on one model. Refinements I asked for were almost always
about **reading**: overlapping titles, legends on top of data, panels too small to see seedlings,
colour clashes between overlays and error maps. The assistant never produced an unreadable chart
twice, but it produced each kind of readability problem once.

**#3 GenAI as a critic.** Used in two ways. First, I had the assistant re-read each figure and print
the numbers behind every claim in its title; this caught an over-claim (crop IoU "does not" drop with
sparsity; it does, only in fewer images) and two wrong numbers in prose. Second, in the revision I
asked for a gap analysis of the submitted notebook against the course description and slides. The
criticism was blunt and justified: the notebook read like a performance report (architecture
table, training recipe, learning curves) and the behaviour part identified patterns without
explaining them. The revised structure (questions and expectations first, then evidence, then a
confirmed / contradicted table) comes from that critique. One part of the critique I did not follow
completely: it proposed to drop the training curves entirely as "not black-box"; I keep the single
observation they support (crop recall is learned in a few epochs, crop precision over the whole run)
as one sentence in the context section, because it motivates the crop/weed question.

**#4 GenAI to steer the investigation.** Each time I shared the current findings, the assistant
proposed the next question. Useful steers: the crop-free images as a probe group (found by the
assistant in the data), "count the crop/weed swaps as objects, not pixels", the perturbation probe
(taken from the slides' list of input-data sources), and the check whether the "false plants" are
unannotated seedlings. Steers I rejected: widening to other encoders or the six-band model, a
training-trajectory re-run, and a Phase II preview. Each was sound and each would have diluted a
single-author Phase I. The breadth of the investigation is clearly larger than I would have reached
alone in the same time; the depth came from my rejections as much as from its proposals.

**#5 GenAI to interpret observations.** Used, and this is where I trust it least. Its first
interpretation of the sparse-weed failure was "part of the measured weed error is annotation
incompleteness", which I found plausible after looking at a few panels and which went into the
submitted notebook. The revision tested it (section 3.3 of the notebook): the isolated "false
plant" blobs are only a few percent of the false-plant pixel mass; the bulk is a thin band of
over-segmentation around annotated plants. The interpretation was not wrong for the one image we
had both looked at, but it was generalised from one image, which is exactly the failure the course
description warns about. Every interpretation in the revised notebook is now tied to a number
computed over the whole test split.

## 2. Answers to the reflection questions of the final-presentation outline

**Did the two pairs pursue different directions?** Not applicable; there was one pair. Compared
with my own plan before the assistant was involved (score tables, a confusion matrix and a few error
panels), the AI-aided work added the per-image and per-object granularity, the confidence analysis
and the perturbation probe, and it removed material that was not about behaviour.

**How did the designs differ / did AI facilitate alternatives?** The assistant usually offered two
or three forms per question (for example histogram versus strip plot for the per-image scores) and
implemented the one I picked. It facilitated implementation far more than ideation: the chart types
are standard, the value was in getting a clean, consistently styled, correctly labelled figure in
minutes and iterating on it. Inappropriate suggestions: figure titles phrased as conclusions before
the data was checked, a "Phase II preview" section that widened the scope to other encoders, and a
first Figure 2 that showed each class in its own frame with the raw blue/green mask colours, which
was harder to read than one overlay per image.

**Did AI contribute to misinterpretations?** Once, the annotation-incompleteness generalisation
described above. It was caught because the revision forced every claim through a computation.

**At which stages was AI most and least helpful?** Most helpful: plumbing and reproduction (finding
the checkpoints on the workstation, matching the training preprocessing bit-for-bit, generating and
executing the notebook, regenerating everything after the folder was lost), and as a tireless
re-reader of figures and numbers. Least helpful: judging what matters for the downstream use. It
treats every pattern as equally interesting; knowing that a crop/weed swap costs more than a
boundary pixel, and that small seedlings are what a sprayer misses, had to come from me.

**Where was human intervention important?** Scope (one model, no training code, no widening),
domain priorities (which errors matter for spraying), verification of the data copy (the swapped
colours), and the final reading of every figure. Also in recovering from tooling accidents: the
kernel that overwrote figures, the folder that was deleted, the shell chains that stopped half-way.

**How did the way of working evolve, and what are the best practices?** At the start I asked for
deliverables ("prepare a handover", "make a notebook"). By the end I asked for *proposals with
alternatives*, decided, and then asked for a *verification pass*. Concretely:

1. Reproduce a known number before building on generated code.
2. Fingerprint data copies; do not trust folder names or prose, including your own documentation.
3. Generate notebooks from a script and execute them headlessly, so prose numbers cannot go stale.
4. Make the assistant re-read its own figures and print the numbers behind each title.
5. Keep one palette and one legend in one module used by everything.
6. Decide scope yourself; accept the assistant's "what next" only when it serves an open question.
7. Keep every deliverable in the repository; side folders get lost.

## 3. Honest assessment of the AI share of this work

The questions, the priorities and all decisions are mine; the data and the model are mine from an
earlier project. The inference scripts, the notebook generator, the figure code, and the first draft
of every document in this repository (including this one) were written by the assistant and then
edited by me. The findings were computed by code that both of us can read and that is in the
repository. I estimate that the assistant roughly tripled what one person could produce in the time
available, at the cost of one generalised-from-one-example interpretation that had to be retracted,
and of several hours lost to tooling accidents.
