# AI interaction record — Phase I (AI-aided pair)

**Team:** AI Gangsta · **Author of this record:** Anton Sadeq (the only active member, see `Group_situation`)
**Tool:** Claude Code (Anthropic), used inside VS Code on the project workspace, 5–24 November 2026.
**Source:** the complete session transcripts are stored locally by the tool. This file is a curated
digest: the prompts that changed the course of the work, what the assistant proposed, what was
accepted, changed or rejected, and why. Prompts are quoted as typed (typos included) and shortened
with "…" where they were long.

## 0. How the collaboration was set up

- The segmentation model and its training pipeline are my own work from an earlier project. The
  training code is deliberately **not** in this repository. The assistant was given read access to
  the pipeline so that the standalone scripts in `scripts/` reproduce the training-time
  preprocessing and scoring exactly, without copying training code.
- The assistant worked as an agent: it could read and write files, run Python, run the model on the
  GPU and copy files from the training workstation over SSH. Every result it reports in this
  repository was produced by code that is in the repository and can be re-run.
- Mode of use (in the terms of the course): mostly *GenAI as collaborator* to design and implement
  the analysis and the visualisations (#2), to steer what to investigate next (#4) and to critique
  figures (#3). *GenAI as tool to transform data* (#1) was used only to build artefacts (predictions,
  tables); no AI-generated labels or judgements enter the analysis. *GenAI to interpret
  observations* (#5) was used with caution, see the reflection.

## 1. Chronological record of the important prompts

### 05.11 — Scoping: what exists, what is needed

| Prompt (abridged) | Assistant's proposal | Decision |
|---|---|---|
| "analyse the current project and find the dataset inside it and the models' artifacts, and tell me what do we need to retrain and have the needed artifacts to work on a AI and Vis project…" | Inventory of the workspace: no checkpoints and no dataset on the laptop, everything on the training workstation; per-epoch logs for 37 runs are local. Proposed: transfer, not retraining. | **Accepted.** Saved days: I had assumed retraining would be necessary. |
| Pasted the course description: "I don't want to share the codes in this project with other team members, so I agreed with them that I will provide them with the needed files/outputs/artifacts…" | A list of artefacts for a black-box Phase I: inputs, masks, argmax and softmax outputs at native resolution, a per-image table, polygons from the raw XML, plus a Phase II re-run with periodic checkpoints. It **assumed** the registered model was the ResNet-101 variant and planned around nine runs. | **Refined.** The artefact list was accepted; the model assumption was wrong (failure pattern F1). |
| "We are not gonna work on the DDR network here…remove any related informations" / "Remove anything about the thesis…" / "only one experiment is enough. The baseline of DLV3+ and MobileNet backbone" | Removed the extra runs, renamed identifiers consistently in 27 files, rewrote the README around one model. While doing so it found that the training log contained three sessions with the same seed and separated them. | **Accepted.** The three-session discovery became a useful reproducibility note (sessions agree within 0.001 mIoU). |
| "analyse the zip files in this directory and check if any of them contains the dataset." | Entry-by-entry scan; none contained data. | Accepted; confirmed my own suspicion. |
| "I am downloading the datasets of beans from the other device … we are only interested in the dataset that was used in this project only." | Defined a **fingerprint** for the right dataset (300 PNGs at 800×1200, split lists 210/30/60, class weights 0.137201 / 1.332565 / 1.0) and a checker script. It rejected two look-alike folders, one of which had crop and weed **colours swapped**. | **Accepted.** Without the fingerprint I would probably have used the wrong copy; see best practice B2. |
| "now beans_segmented is being transferred." | Background watcher, then full verification of the raw export (300 samples, band count, mask colours, polygon labels), rebuild of the RGB dataset with the project preprocessing, regeneration of splits and class weights — all identical to training. It also noticed that 27 images contain no crop and proposed them as a probe group. | **Accepted.** The crop-free group became question Q2b in the notebook. |

### 06.11 — Preparing the feedback session

| Prompt | Assistant's proposal | Decision |
|---|---|---|
| "I have a feedback session with the professor now, what questions should I ask her…" | A grouped list: whether private training code is acceptable, whether one model is enough, what "behaviour" means for a dense-prediction model (per image, per pixel, per object), whether perturbations are expected, whether test may be analysed in Phase I, which attribution methods fit segmentation. | **Used as the agenda.** The answers were not recorded in the transcript; the notebook follows the course slides where they were explicit (perturbations and the training data as hypothesis sources, object-level analysis). |

### 10.11 — Inference scripts and first results

| Prompt | Assistant's proposal | Decision |
|---|---|---|
| "prepare scripts for inferring the models … calculating the predictions of the models on the test subset … give me the results of the models on the test set … so that my colleagues can experiment the models on the testing set." | Located the checkpoints on the workstation, pulled them, re-implemented the evaluation preprocessing in plain OpenCV (verified against the training transform to 2.4e-7), wrote `predict.py` / `evaluate_predictions.py` / `export_onnx.py` / `inspect_model.py`, produced predictions for all 300 images, and reproduced the archived test score to 2e-5. It reported two scorings (network grid vs original grid) because they differ by 0.003. | **Accepted.** I insisted on the exact reproduction of the archived score before anything else was built on the predictions. |
| (same session, unprompted) | Added a note after looking at error panels: on sparse-weed images many "false plant" pixels sit on unannotated seedlings. | **Accepted as a hypothesis**, later became the annotation-incompleteness analysis (Q2c). |

### 11.11 — Midpoint notebook (deadline day)

| Prompt | Assistant's proposal | Decision |
|---|---|---|
| "I want to submit something in this submission from the whole system we have here. maybe we make half submission, what can be easily done here?" | A list of figures computable from the existing CSVs and masks in about an hour (score table, confusion heat-map, per-image mIoU strip plot, IoU vs pixel count, confidence vs error, error panels, training curves) and a list of things to state as **not done** rather than fake (perturbations, object-level scoring). | **Accepted**, with one change: I chose a notebook over a PDF. |
| "the submission can be a notebook or a pdf. I would prefer to have it as a jupyter notebook. … just work." | Generated the notebook from a script (`tools/build_notebook.py`), executed it, exported HTML, nine figures. Visualisation design decisions by the assistant: strip plot with dataset-level line and per-image median annotation; log-x scatter of IoU vs pixel count; horizontal stacked bars splitting errors into boundary and interior; reliability bars plus a cumulative "errors caught" curve; 2×2 error panels. | **Accepted.** Two figure titles were corrected after the assistant checked them against the plotted data (F3). |
| Screenshot of Figure 9 (panels too small, title overlapping) | Resized the figure and moved the title. | Accepted. |
| "I want to improve this. make the crop green and the weeds red. you don't have to plot each class in a separate frame." (Figure 2) | Changed the overlay palette **package-wide** rather than only in the figure, moved the crop/weed swap colour in error maps from red to cyan to avoid a clash, regenerated all panels. | **Accepted.** The design request was mine; the consistency decision was the assistant's. |
| (incident, same evening) | My open notebook kernel had imported the old palette and re-ran cells, overwriting the figures with the old colours twice. The assistant diagnosed it and added an explicit module reload to the setup cell. | Fixed (F2). |

### 24.11 — Revision of the submission

| Prompt | Assistant's proposal | Decision |
|---|---|---|
| "I have now the chance to modify the mid submission … my group mates left me alone … define how to modify this submission in our favour, what more can we do? what to avoid? how to make it simple? … You can only refer to the slides and the description of the mid submission as references" | Gap analysis against the course description: two named deliverables were missing (this record and the GenAI reflection); the notebook was too much "performance report" (sections on architecture, recipe, training curves) and too little "behaviour in depth"; proposed to cut those sections, add an object-level analysis of the sparse-weed failure, an annotation-incompleteness check, a confidence-by-error-type view and a perturbation probe, and to restate the single-author situation plainly. | **Accepted.** |
| "Forget about the handover folders. we don't need to handover anything to anyone, we have already everything here." | Stopped the handover framing; proposed to make the GitHub repository itself the deliverable folder. | Accepted. |
| "First we should make a sub-directory in this directory here and clone the github repo here, and then add there all the necessary codes, artifacts, and notebooks, or visuals…" | Cloned the repository, laid out `model/ scripts/ run_metadata/ splits/ notebooks/ predictions/ tools/`, re-pulled the dataset and the last checkpoint from the workstation, regenerated every prediction and the ONNX export, re-executed the notebook, git-ignored the 640 MB of images and the per-pixel outputs. | Accepted. |
| "Let's go, the course slides you can find under: course_materials" | Read all slide decks, then restructured the notebook into the structure the final-presentation outline asks for (problem context → questions and expectations → investigation and findings → reflection), wrote this record and the reflection, rewrote the README, added `tools/run_perturbations.py` and the object-level analysis. | Accepted; this is the state of the repository. |

## 2. Failure patterns observed

- **F1 — Confident wrong assumption about scope.** The first plan assumed the registered model was the
  ResNet-101 variant and planned a nine-run comparison. The assistant stated the assumption, which is
  why it was caught in one reply, but had I not read carefully the whole handover would have been
  built around the wrong model. *Lesson:* state the registered model and the scope in the first
  prompt of every session.
- **F2 — Two writers on one file.** AI-generated code changed a shared module while my own notebook
  kernel still had the old version imported; re-running my cells silently overwrote figures with the
  old colours. Not an AI error as such, but a workflow hazard of agentic tools that edit files you
  have open. *Lesson:* restart kernels after the assistant touches shared code; let the assistant
  put explicit reloads into setup cells.
- **F3 — Over-claiming figure titles.** A first title of the IoU-vs-pixel-count figure claimed crop
  IoU was unaffected by sparsity; the data showed crop drops too, only in fewer images. The assistant
  corrected it when it inspected the rendered figure. *Lesson:* every figure title is a claim; have
  the assistant re-read its figures and print the numbers behind each claim.
- **F4 — Numbers written before they were verified.** In an early README draft two values
  (a parameter count and the epoch of minimum validation loss) were wrong; the assistant's own
  verification pass found and fixed them. *Lesson:* ask for a verification pass against the logs
  for every number in prose, or generate prose numbers from code (the notebook does this).
- **F5 — Scope creep by helpfulness.** Repeatedly the assistant proposed widening (all encoders, the
  six-band variant, a training-trajectory re-run). Each was reasonable and each would have cost
  days. *Lesson:* decide the scope yourself; a single model studied deeply is what the course asks.
- **F6 — Environment fragility.** Chained shell commands aborted half-way when one step failed, a
  disk-usage command hung on a network drive, API outages dropped two prompts, and a context reset
  in a long session lost the assistant's knowledge of where files were. Nothing was lost in the end
  because everything important had been written to files, but time was.
- **F7 — Stale documentation taken at face value.** A quick-reference line in my own pipeline README
  claimed the splits were stratified on weed presence; the assistant discovered by regeneration that
  the script default (crop presence) was what actually ran. *Lesson:* regenerate, do not trust
  prose, including your own.

## 3. Emerging best practices

- **B1 — Reproduce the known number first.** Nothing was built on the predictions until the scripts
  reproduced the archived test mIoU to 2e-5. Every later finding rests on that check.
- **B2 — Fingerprint the data.** A short checker (image count, size, split lists, recomputed class
  weights) caught a near-identical dataset copy with swapped class colours.
- **B3 — Generate the notebook from a script.** `tools/build_notebook.py` writes the notebook; the
  notebook is executed headlessly. Figures, tables and the numbers in the prose come from the same
  run, so a re-execution cannot leave stale text behind.
- **B4 — Fix conventions package-wide.** One palette, one error-colour legend, one class order,
  defined in `scripts/beans_common.py` and used by the panels, the notebook and the README.
- **B5 — Let the assistant propose, decide yourself, then let it verify.** The best interactions
  had three steps: a proposal with alternatives, my decision (often narrowing), and a verification
  pass in which the assistant checked its own output against data or files.
- **B6 — Keep deliverables in the repository, not in a side folder.** The deleted handover folder
  cost an afternoon of regeneration; the repository with a `.gitignore` for the heavy files is the
  only copy now.
- **B7 — Write down what was not done.** On deadline day the assistant listed what to state as "not
  yet done" instead of faking it; the revised notebook keeps that habit (section 6).

## 4. Who contributed which idea

| Idea | Origin |
|---|---|
| Study one model (DeepLabV3+ MobileNetV2) in depth; no handover of training code | me |
| Artefact list for a black-box Phase I (native-resolution masks, softmax maps, per-image table, two scorings) | assistant |
| Crop-free images as a probe group; three training sessions as a reproducibility note | assistant (found in the data / logs) |
| Notebook as the submission format; crop green / weed red; one frame per example | me |
| Per-image strip plot against the dataset-level score; IoU vs log pixel count; boundary/interior error split; calibration + "errors caught" curve; error panels | assistant |
| Annotation-incompleteness hypothesis on sparse-weed images | assistant (from looking at panels); confirmed by me on the raw images |
| Perturbation probe, object-level detection analysis, confidence by error type, vegetation-index check of "false plants" (24.11 revision) | assistant, from the course slides' list of input-data types; scope approved by me |
| What to cut (architecture table, recipe, training curves, Phase II widening) | assistant proposed, I decided |
