# CLAUDE.md: PA3 master instruction prompt (carried over from PA2)

This file is a **template**, written before the PA3 manual exists. It carries over every workflow rule,
protocol and lesson from PA2 that is not PA2-specific. Sections marked **`[FILL IN FROM PA3 MANUAL]`**
are placeholders: read the real PA3 manual first and replace them before starting any work. Everything
else is a standing process rule that applied throughout PA2 and should apply again unless the PA3
manual explicitly says otherwise.

---

## 0. Standing decisions (override anything below that conflicts)

- **Authority**: the released PA3 manual/PDF is the primary spec. Priority order: manual > experimental
  validity > reproducibility > correctness > code organization > result collection > factual analysis >
  convenience. Manual: `ATML-PA3 (1).pdf` in the repo root (released 23 Oct 2026, due 06 Nov 2026)..
- **Authorship of code**: Claude writes the implementation code directly (scripts or notebooks, matching
  the released starter's own style per task), after the user approves the plan for that task.
- **Starter code is not a reference solution**: if PA3 follows PA2's pattern, the starter scaffolds may be
  incomplete and/or contain deliberate defects in core objective implementations. Before trusting any
  output: validate the released implementation against the mathematical/algorithmic objective stated in
  the manual, identify and fix any defect, explain the fix and the reasoning, and get user approval before
  proceeding if the fix changes experimental behaviour. **Never silently patch around a bug.**
  PA3 ships **no starter code** (manual + supplied Task 3 rollout data only), so there are no deliberate
  starter defects; all code is written from the manual. Validate against the manual's formulas instead.
- **Environment setup**: use the starter repo's own pinned `requirements.txt` / environment file and its
  own documented install steps. Confirm any provided validation/setup script passes before starting task
  work. Setup: see README §1 (`uv venv --python 3.12`, `uv pip install -r requirements.txt --index-url
  https://download.pytorch.org/whl/cu128 ...`, then `python -m scripts.check_environment` and `python -m scripts.prefetch`).
- **Model/policy family** (if applicable): Qwen/Qwen3-0.6B (`enable_thinking=False`) for Tasks 1-2;
  google/siglip-base-patch16-224 (frozen) for Task 2; Qwen/Qwen2.5-3B-Instruct rev aa8e7253... for Task 3 live.
  LoRA/OPLoRA implemented in `common/lora.py`; all fixed hyperparameters live in `configs/task{1,2,3}.yaml`.
- **Git discipline**: never run `git commit` or `git push` without being explicitly told to. Stage and
  review changes, then hand the user the exact commands to run themselves. Before any destructive git
  operation (`checkout`/`restore`/`reset`/`clean`, etc.), run `git status` first and stash or commit
  anything uncommitted. Before a broad `git add`, review `git status` for anything that shouldn't be
  staged (checkpoints, secrets, large binaries).
- **Assets discipline**: large checkpoints/caches/datasets come from wherever the course provides them
  (asset repo, download script, etc.), never hand-authored or committed. Don't alter or commit anything
  under directories marked as supplied/read-only (e.g. `cached/`, `checkpoints/`). Never commit raw
  datasets, public model weights, or unnecessary large checkpoints.
- **Seed**: fix a seed (PA2 used 6304) wherever the manual allows a choice ("fix random seeds where
  practical"). PA3's manual itself specifies seed 6304 for every task.
- **Fixed files — never regenerate**: any course-supplied CSV/JSON/JSONL that defines a fixed split,
  prompt pool, diagnostic set, or similar must be used unchanged once released. Identify these explicitly
  at the start of each task and never overwrite them. Fixed files (written once by `save_fixed_file`, never regenerate):
  `data/task1/train_ids.json`, `data/task1/probe_set.json`, `data/task2/manifest.json`,
  `data/task2/grounding_pools_train.json`, `data/task3/splits.json`, `data/task3/orders.json`, plus the external
  inputs `data/task1/first_plot_questions.yaml`, `data/task3/mathvista_testmini.json`, `data/task3/raw/*`.
- **Report**: maintain a detailed, factual **semi-report** (working reference, not the final submission)
  throughout, named e.g. `results/report.md`, git-ignored. See §11 for the critical distinction between
  this semi-report and the actual submitted PDF, which has AI-authorship restrictions — **read §11 before
  writing anything that could become submission text.**

---

## 1. Workflow: one task at a time

Work through the assignment's tasks in the order the manual specifies. **Never implement several tasks
at once.**

**Before each task** (don't create/modify files beyond reading the manual before explicit approval):
1. Re-read the task's section of the manual.
2. Summarize requirements, steps, required evidence and research questions.
3. Propose the directory/file structure (diffing against the released starter, if one exists).
4. List files to create/complete.
5. Explain the objective/architecture, including what in the starter is a scaffold versus a
   known-defective implementation to fix.
6. Explain the experimental protocol (fixed budgets, prompts/data, decoding or other fixed settings).
7. Identify leakage/reproducibility concerns and ambiguities.
8. Ask permission.
9. Wait.
10. Implement only after approval.

**After each task**:
1. Syntax checks and smoke tests.
2. Show the files created/completed, including any objective-defect fix and a brief correctness argument.
3. Explain the implementation.
4. Give exact run commands.
5. Run only approved experiments, at the fixed budgets the manual specifies.
6. Collect results.
7. Save machine-readable results.
8. Save figures.
9. Update the semi-report.
10. Summarize.
11. **STOP.**
12. Ask whether to proceed to the next task.

If the manual is ambiguous, **stop and ask** before making a consequential decision. Don't guess silently
on anything that would change experimental behaviour or results.

---

## 2. Repository structure

Follow the released starter's layout if one exists; otherwise propose a clean one and get approval before
creating it. Keep task boundaries explicit and put only genuinely shared code in a common/shared module.
Each task must be runnable without executing previous tasks interactively; save intermediate
generations/metrics to disk whenever later analyses reuse them — never rely on hidden notebook state.

PA3 layout (no starter; created in session 1): `common/` (utils, lora, plotting), `task1_finetune/`,
`task2_vlm/`, `task3_ttc/`, `scripts/` (check_environment, prefetch, leakage_check), `configs/task{1,2,3}.yaml`,
`run_overnight.sh` (resumable orchestrator), `results/task{1,2,3}/`, `checkpoints/` (git-ignored).

---

## 3. Results logging protocol

Whenever a run finishes, extract and save its results immediately. Don't rely on chat history for numbers
— if it isn't on disk, it doesn't exist for the purposes of the report.

1. **Primary layout**: results and generated outputs under a per-task results directory, descriptive file
   names.
2. **Consolidated JSON**: one per task, nested keys mirroring experiment structure, full
   per-update/per-epoch trajectories where applicable (reward, KL, losses, entropy, clip fraction, grad
   norm, response length, or whatever the task's analogous metrics are), raw unrounded floats.
3. Once all tasks have results, merge into one all-results file.
4. Metric not computable or run failed: store `null` plus a note explaining why — **never omit silently**.
5. After logging, print a short summary (paths + key numbers).
6. Machine-readable formats (json/csv/yaml/txt). Figures as PNG (+PDF if a report will need vector
   figures). Never screenshots only.
7. Don't overwrite results accidentally: replace only after confirmation, or save as a new run.
8. Every training/experimental run records: seed, exact prompt/data IDs or indices used, config path,
   checkpoint path (if applicable), optimizer/LR/schedule, batch/group size, update/epoch budget,
   hyperparameter values under study, decoding settings used for evaluation (if applicable), and final +
   trajectory metrics. Save training curves.
9. Plots need meaningful axis labels, legends, units and readable fonts.
10. Record peak VRAM and wall-clock time for every substantial run, if the manual asks for compute
    evidence (it did in PA2 for the standard PPO/GRPO continuations).

---

## 4. Never fabricate

Never invent any metric, training curve, timing, or qualitative example. Not run: `Status: NOT RUN`.
Failed: `Status: FAILED` + reason, error, what was attempted, what's unresolved. This applies to every
number that ends up in the semi-report or (later, in the student's own words) the final report.

---

## 5. Reproducibility

Fix the seed wherever the manual allows a choice. Preserve and save the exact prompt IDs/data indices,
decoding configuration (fixed across conditions being compared), maximum generation length (if
applicable), seed, hyperparameters, optimizer/adapter config, and evaluation procedure. Save the actual
config behind every major result. Never silently change a setting the manual fixes; don't choose a final
setting after inspecting held-out results unless a task explicitly asks for that study.

---

## 6. Leakage / no-retuning rules (critical — fix before proceeding if violated)

- Use the **same fixed prompt/data set, decoding configuration, max generation length, seed and
  evaluation procedure** across every condition being compared within a study.
- If the assignment has an analogue of PA2's Task 4/5 (evaluation-only tasks that run after earlier
  training is frozen), their results must never be used to retune earlier training, pick a different
  checkpoint, or change hyperparameters.
- Do not change thresholds, parsers, or evaluation settings after seeing results on a held-out/fixed
  evaluation file. Fixed files are used unchanged.
- No cherry-picking, no discarding inconvenient failures. If results are surprising, investigate; don't
  alter the protocol to make them look better.
- Run a leakage check (exact prompt/text overlap between every training file and its evaluation file) once
  all data is assembled, and record the result (PA2 did this via a small post-hoc script).

PA3 specifics: Task 1 excludes MetaMathQA rows matching GSM8K test questions; evaluation results never
feed back into training. Task 2: eval images (val2014) never in training/replay; never select checkpoints with
final-test results; label-rule changes only before training. Task 3: selection/stopping rules may use neither gold
answers nor unrevealed candidates; adaptive thresholds chosen on the 200-question dev split only, then frozen.

---

## 7. Task requirements

Task 1: Qwen3-0.6B on 40k MetaMathQA, five variants (Full FT, L2-SP lambda=1e-3, LoRA r4/r16, OPLoRA r16 k128),
1,250 updates each; lm-eval GSM8K/HellaSwag/BoolQ/ARC-Easy on seven conditions; weight-space and activation
diagnostics; 168 qualitative responses (42 annotated). Task 2: SigLIP-Qwen VLM, 2,250/1,500/750-update curriculum,
grounding evaluation under correct/swap/mismatch/no-image with pair bootstrap. Task 3: offline Best-of-N /
self-consistency / RWC / VisConf on supplied rollouts, live parallel vs sequential refinement (50 MATH problems),
adaptive stopping. Full summary and implementation decisions: README §5; status: `results/report.md`.
The user asked (session 1) for all tasks' code at once plus an overnight script, overriding the
one-task-at-a-time flow for implementation; experiments still need explicit go-ahead.

---

## 8. Expensive runs

Before a substantial run, state what will run, the expected cost, disk/VRAM use, the files produced,
whether a GPU is needed, and the exact command, then ask. Smoke-test first. Small tests and syntax checks
can run automatically. Any fixed budgets the manual specifies (update counts, epoch counts, etc.) are not
something to shorten or extend without approval.

---

## 9. Code quality

Modular, readable, config-driven, relative paths only (no machine-specific paths), no hidden
hyperparameters or magic numbers, no duplicated implementations. Attribute any materially reused external
code in the README (source, URL, what was reused, what was modified) — this is a hard submission
requirement in most course manuals, not just a convention.

---

## 10. Debugging

Reproduce → identify the exact cause → explain it → propose a fix → get approval if the fix changes
experimental behaviour → rerun the smoke test → record the fix. This governs both ordinary bugs and any
deliberate per-task objective defects the manual warns about: validate against the mathematical/algorithmic
objective, document what was wrong and why, and never silently change methodology just to make code run.

---

## 11. Final report boundary — read this before writing any report text

PA2's manual had an **AI usage policy**: generative AI could not be used to write any part of the PDF
report; the report's language, interpretation and analysis had to be entirely the student's own. Coding
assistance from an LLM was separately permitted and had to be attributed if materially reused. **PA3 has
the same policy (checked): no generative AI in any part of the PDF report; coding assistance allowed with
attribution in the README.** It was confirmed before any report drafting, and if it has an
AI-authorship restriction on the report text, apply it from the start rather than discovering the conflict
after a full draft is written (that happened in PA2 — don't repeat it).

Under a PA2-style policy, Claude's role is:
- Maintain `results/report.md` (or equivalent) as a **detailed, factual semi-report**: every number
  traces to a saved result file, organized per task, with tables/figures and plain factual statements of
  what the data show. This is a *reference the student writes their own report from*, not itself
  submittable text.
- Do **not** draft full prose paragraphs intended to be copied into the submitted PDF — that is exactly
  the kind of "report language, interpretation and analysis" the policy reserves for the student. If asked
  to "write the report" in a way that would become submission text, say so explicitly and ask whether the
  output is meant as (a) a factual semi-report/reference the student will rewrite in their own words, or
  (b) something else — don't silently produce final-looking prose either way.
- If the student nonetheless asks for a full draft report (as happened in PA2), it's reasonable to help,
  but **flag the policy conflict explicitly** rather than letting it pass unmentioned, so the student can
  make an informed call about what they submit.

If PA3's policy differs (e.g. permits AI-assisted writing, or has no such restriction), update this
section accordingly once the manual is read, and proceed normally.

### Report structure and style (reusable mechanics, independent of the AI-authorship question)

These are mechanical/organizational lessons from PA2, reusable regardless of who ends up authoring the
submitted prose:

- **Heading structure should mirror the assignment's own task structure** (one section per task, in the
  assignment's order), not a generic suggested outline, unless the student says otherwise. PA2's manual
  suggested one organization; the student preferred headings that followed *their own* per-task structure
  (Setup → Experiments → Results → Analysis per task) instead.
- **Don't force a separate "Research Questions" section** if the student doesn't want one — research
  questions can be answered inline inside the relevant discussion paragraph, each flagged by a short
  lead-in phrase ("This answers the first research question: ...") rather than a boilerplate "RQ1./RQ2."
  block, if that's the student's preference. Confirm the preference; don't assume.
- **Page limits are real constraints.** If the manual states an 8-page limit (references/appendix
  excluded), track page count continuously while drafting, not just at the end. Condensing later is much
  more work than writing tightly from the start. When asked to cut to a target page count:
  - Cut paragraph-by-paragraph, verifying the page count after each real edit (not after cosmetic changes
    alone) — floats can shift unpredictably, so re-measure, don't assume.
  - Preserve every number, caveat, and research-question answer; cut redundant restatement, merged
    setup/metrics sub-headings into flowing prose, and dropped figures/tables whose content is already
    stated in the text.
  - Flag clearly which required-evidence items (if any) ended up covered only in prose instead of a
    dedicated figure/table, so the student can decide if that's acceptable.
  - **Never edit the student's live/source file in place when doing exploratory trims** — work on a
    scratch copy, verify it builds and fits the page count, and only then ask whether to apply the change
    to the real file. (This was a late correction in PA2's workflow: trimming should default to a copy
    unless told to edit the source directly.)
- **Style conventions that held throughout PA2** (apply by default unless the student's own style
  guidance says otherwise):
  - No em dashes or en dashes anywhere in report prose — use "to", "and", commas, or rephrase.
  - Percentage-point differences in rates get "pp", not bare "%", when distinguishing a rate's own change
    from a relative percentage.
  - Mark untested explanations as hypotheses explicitly ("untested", "possibly", "a plausible but untested
    mechanism is...") — never present a hypothesis as a finding.
  - Every reported number must trace to a result file; never invent or silently re-round a number
    differently from its source.
  - Figures/tables are referenced from the text near where they're first discussed, not dumped at the end.
  - Plain, human-sounding academic prose; avoid formulaic sentence templates repeated verbatim across every
    paragraph (a known failure mode — e.g. ending every single paragraph with the exact same phrase).

### LaTeX/PDF build pipeline (if the report needs to be NeurIPS-style or similar)

PA2 used a from-scratch Markdown → LaTeX converter rather than writing LaTeX by hand, because it let the
student keep editing a single Markdown source while Claude (re)generated the `.tex`/PDF on request. If PA3
needs something similar:
- Keep one canonical Markdown source file. Never regenerate it from parts; edit it in place with targeted
  edits, and **re-read it before editing** in case the student edited it by hand outside the session.
- Write a small, documented converter script (not committed as a hidden dependency the student can't
  inspect) that understands: a title block before `## Abstract`; the abstract between `## Abstract` and
  `## 1.`; `## N. Title` / `### N.N` / `#### N.N.N` headings (numbers stripped, LaTeX auto-numbers them);
  figures as `![Figure N. caption](path)` on their own line; tables preceded by a bold `**Table N.
  caption**` line; display math `$$...$$` and inline `$...$`; a `## References` heading that triggers
  `\clearpage` + bibliography. Extend it for other conventions (citations, appendices) only as the student
  actually uses them, rather than guessing ahead.
- Use the course-provided style file (`.sty`) if one exists; don't invent a competing style.
- Build with the standard four-pass sequence (`pdflatex`, `bibtex`, `pdflatex`, `pdflatex`) and always check
  the log afterward for errors, undefined references/citations, overfull/underfull boxes, and stray control
  characters before declaring the build done.
- Only rebuild the PDF when asked — it costs tokens and time; don't rebuild speculatively after every
  small text edit unless the student wants to see each iteration.
- Don't touch the submitted PDF/`.tex` as a side effect of something else — if the student asks for the
  PDF/`.tex` specifically, apply changes only to those build artifacts (from a copy or the source, as
  instructed) and say explicitly what was or wasn't touched.

---

## 12. Task completion checklist (adapt the specifics once PA3's manual is known)

**Code**: files exist, structure follows the manual/starter, runs from the repo root, any per-task
objective defect is found and fixed (with justification), configs saved, seed fixed, no leakage/
no-retuning violation.

**Experiments**: all required steps, ablations, fixed budgets, and controlled studies done at the
specified settings.

**Results**: machine-readable files + figures exported to the results directory; every required-evidence
item present and every number traceable.

**Analysis**: semi-report updated, every research question has evidence listed, contradictory evidence
recorded, nothing fabricated.

---

## 13. General collaboration notes carried over from PA2

- Confirm before destructive or hard-to-reverse actions (git history rewrites, overwriting results,
  deleting files) unless already durably authorized.
- When uncertain whether the user wants a quick answer or a deliverable, lean toward asking once rather
  than guessing and producing the wrong shape of output (e.g. full prose draft vs. factual reference).
- State assumptions plainly when the manual is silent on something minor, but stop and ask for anything
  that would be consequential or hard to undo (training budget, which checkpoint counts as "standard",
  whether a result is in-scope for a given task).
- When something is done and verified, say so plainly without hedging; when a step was skipped or a test
  failed, say that plainly too — don't round failures up to successes.
