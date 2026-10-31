# ATML PA3: fine-tuning, visual grounding and test-time compute

Code for the three PA3 tasks: (1) Qwen3-0.6B mathematical fine-tuning with Full FT, Full FT + L2-SP, LoRA-4,
LoRA-16 and OPLoRA-16; (2) a SigLIP-Qwen3 VLM trained through a three-stage curriculum and evaluated for visual
grounding; (3) test-time compute on the supplied Qwen2.5 / Qwen2.5-VL rollouts plus a live MATH experiment.
All settings follow the PA3 manual; every deviation or implementation choice is listed in
[Implementation decisions](#implementation-decisions-and-deviations).

## 1. Setup

Python 3.12, CUDA GPU (developed on an RTX 4060 Laptop, 8 GB). From the repository root:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
.venv/Scripts/python.exe -m scripts.check_environment     # (.venv/bin/python on Linux)
.venv/Scripts/python.exe -m scripts.prefetch              # models, datasets and external inputs
```

`requirements.lock.txt` holds the exact resolved versions (torch 2.11.0+cu128, transformers 4.57.1,
lm-eval 0.4.13, math-verify 0.9.0). `scripts.prefetch` downloads everything that is not committed: Qwen3-0.6B,
SigLIP-base-patch16-224, Qwen2.5-3B-Instruct at the README-pinned revision, MetaMathQA, GSM8K and the
lm-eval task data, the supplied ATML-PA3-Data rollouts (Google Drive), MathVista `testmini.json`,
`first_plot_questions.yaml`, and the COCO 2014 / VQAv2 annotation archives. COCO images are fetched
individually (only the ~9,200 selected ones) by `task2_vlm.data_prep`.

## 2. Running everything (overnight)

```bash
./run_overnight.sh          # setup -> Task 3 -> Task 2 -> Task 1 -> leakage check   (Git Bash on Windows)
./run_overnight.sh task1    # or one phase: setup | task3 | task2 | task1 | final
```

The runner is resumable: finished steps are skipped and interrupted training resumes from the latest recovery
checkpoint, so rerun it each night until `results/_overnight/STATUS.txt` ends with `ALL DONE`. Per-step logs
are in `results/_overnight/logs/`. A failed step is retried once; steps that depend on it are skipped while
independent steps continue. After the setup step the run uses `HF_HUB_OFFLINE=1`.

Measured / estimated wall clock on the RTX 4060 Laptop GPU:

| Phase | Main cost | Estimate |
|---|---|---|
| Task 3 | offline analysis ~3 min; live generation (50 questions, ~10 tok/s for the 3B FP16 model) | ~3.5-5 h |
| Task 2 | feature caching ~3 min; 4,500 updates at ~0.9 s; evaluation ~3 min | ~1.5-2 h |
| Task 1 | five 1,250-update runs (Full FT ~11-16 s/update with optimizer offload, LoRA ~10 s, OPLoRA ~12 s); seven lm-eval passes (~30-60 min each) | ~25-30 h |

Plan for roughly three nights. Keep the laptop plugged in and stop Windows from sleeping during the run
(for example `powercfg /change standby-timeout-ac 0`, restored afterwards). Disk: Task 1 recovery checkpoints
use ~60 GB (Full FT states are 7.2 GB each); Task 2 features ~2.8 GB.

## 3. Per-task commands

Every entry point takes `--config configs/taskN.yaml` (the default) and most accept `--smoke` for a tiny
test run that writes to `*_smoke` directories.

**Task 1** (`task1_finetune/`)
```bash
python -m task1_finetune.data                         # fixed 40,000-example ID file + tokenized cache
python -m task1_finetune.probe_set                    # fixed 32 + 32 activation-probe set
python -m task1_finetune.train --method full          # full | l2sp | lora4 | lora16 | oplora16
python -m task1_finetune.merge --method lora16        # adapters -> fresh backbone, merged-vs-unmerged check
python -m task1_finetune.evaluate --condition full    # baseline | full_mid | full | l2sp | lora4 | lora16 | oplora16
python -m task1_finetune.qualitative --condition full # 24 greedy responses; --export builds the annotation sheet
python -m task1_finetune.diagnostics                  # rho_k, eta, R(z) for down_proj layers 0/13/27
python -m task1_finetune.summarize                    # tables, figures, results/task1/task1_results.json
```

**Task 2** (`task2_vlm/`)
```bash
python -m task2_vlm.data_prep            # fixed manifest + grounding pools, image download (--inspect: label sheets)
python -m task2_vlm.features             # SigLIP feature cache (FP16), encoder released afterwards
python -m task2_vlm.checks               # feature shapes, frozen sets, connector gradients, split isolation
python -m task2_vlm.train                # 3-stage curriculum (--variant pool49 for the optional 49-token run)
python -m task2_vlm.evaluate             # final four-condition evaluation of the Stage 3 model
python -m task2_vlm.summarize            # learning curves, final tables, qualitative pairs
```

**Task 3** (`task3_ttc/`)
```bash
python -m task3_ttc.data       # join + alignment checks, fixed splits/orders, answer groups and scores
python -m task3_ttc.offline    # Steps 1-3: budgets, selection rules, failure decomposition, signal analysis
python -m task3_ttc.adaptive   # Step 5: development-selected stopping rules, held-out evaluation
python -m task3_ttc.live       # Step 4: single vs parallel vs sequential refinement (GPU)
python -m task3_ttc.summarize  # Step 6: figures and tables
```

`python -m scripts.leakage_check` verifies train/evaluation isolation for all tasks.

## 4. Outputs

| Path | Content |
|---|---|
| `data/task1/train_ids.json`, `probe_set.json` | fixed MetaMathQA source indices; probe IDs and prompt strings |
| `data/task2/manifest.json`, `grounding_pools_train.json` | every selected image / question / pair ID, mismatched-image mapping, training pools |
| `data/task3/splits.json`, `orders.json` | dev / held-out / live question IDs; ten rollout orders per question |
| `results/task1/` | per-update logs, run summaries (time, memory, parameter counts, checkpoint sizes, overheads), per-example lm-eval outputs, merge checks, diagnostics, qualitative responses and annotation sheet, tables and figures, `task1_results.json` |
| `results/task2/` | data stats, feature-caching cost, checks, train and dev logs, final predictions and metrics with bootstrap CIs, label-inspection sheets, figures, `task2_results.json` |
| `results/task3/` | per-rollout answer groups, per-(question, order, budget, method) selections, coverage, adaptive decisions, live calls, tables, figures, `task3_results.json` |

Fixed files are written once (`save_fixed_file` refuses to overwrite) and must not be regenerated.

## 5. Implementation decisions and deviations

Settings fixed by the manual are used as specified. The following choices were needed where the manual is
silent or the hardware forced an implementation detail; none changes an objective or a fixed budget.

**Task 1**
- *Full FT optimizer offload.* FP32 master weights + FP32 gradients + FP32 AdamW moments (~9.5 GB) exceed the 8 GB
  GPU. AdamW moments are kept in CPU RAM and streamed per tensor chunk (`task1_finetune/optim.py`). The update is the
  exact `torch.optim.AdamW` rule (verified: max difference 0.0 on random problems). Its time is reported as Full FT
  overhead.
- *L2-SP.* theta_0 is held on the CPU in BF16, which is exact for the BF16 Qwen3 release (checked element-wise at
  start-up, with an FP32 fallback). lambda (theta - theta_0) is added to the unscaled gradient before clipping, which
  equals the gradient of SFT + lambda Omega. Omega sums all unique parameters (tied embedding counted once).
- *Loss computation.* The LM head and cross-entropy are evaluated only at supervised positions, in checkpointed chunks
  of 256 tokens, and autocast's weight-cast cache is disabled. Both keep memory within 8 GB without changing the
  mathematics. The loss for each update is the summed response NLL over its 32 examples divided by the number of
  valid response tokens in that effective batch.
- *Skipped updates.* With FP16 dynamic loss scaling the first update(s) overflow at the default initial scale (65,536);
  these are logged in `skipped_updates`. The LR schedule advances once per update attempt (1,250 in total).
- *Cosine schedule* decays to zero after the 38 warmup updates (`transformers` convention).
- *OPLoRA.* Singular bases are computed once in FP32 with the CPU (LAPACK) SVD: the CUDA FP32 SVD returned bases
  orthonormal only to ~7e-4, which visibly weakened the projection. Projected factors are always computed in FP32,
  once per microbatch (reused by gradient-checkpoint recomputation; gradients identical to recomputing them per call),
  and exported for merging and diagnostics. Measured constraint residual ||U_k^T dW|| / ||dW|| ~ 3e-7.
- *LoRA* is implemented directly (`common/lora.py`): A ~ Kaiming-uniform (PEFT default) from a seeded CPU generator in
  module order, so LoRA-16 and OPLoRA-16 start from identical unprojected factors; B = 0; dropout 0.05 on the input.
- *Microbatch.* Training is overhead-bound at microbatch 2 on this GPU. A larger microbatch with the same effective
  batch would be mathematically equivalent here (the loss is normalized over the whole effective batch) and roughly
  2-4x faster, but the manual fixes microbatch 2, so `configs/task1.yaml` keeps 2.
- *GSM8K truncation flag.* lm-eval strips stop sequences and does not expose generated-token counts, so a response is
  flagged as truncated when its re-tokenized length is within 4 tokens of the 1,024 cap.
- *GSM8K overlap filter.* Normalization = NFKC, lowercase, non-alphanumerics to spaces, collapsed whitespace. 0 of the
  first 40,002 eligible MetaMathQA rows matched (MetaMathQA is derived from GSM8K/MATH training splits); 2 rows were
  excluded for length.

**Task 2**
- *Prompt text.* Caption instruction "Write a short caption for this image."; questions are followed by
  "Answer the question using a single word or phrase." (LLaVA-1.5 convention), identical in training and evaluation.
  Caps (80 question / 64 caption / 16 answer tokens) truncate. Visual tokens are `<|vision_start|>` + 196 x
  `<|image_pad|>` + `<|vision_end|>` + newline before the question inside the user turn.
- *Grounding labels* use bounding-box area fractions: presence positives need a non-crowd instance >= 1% of the image,
  negatives have no annotation of the category; counting needs 1-4 non-crowd instances, no crowd annotation and every
  instance >= 0.5% of the image. Evaluation pairs are half presence and half counting. Label sheets
  (`results/task2/label_inspection/`) allow checking labels by eye before training. COCO labels are visibly noisy for
  some counting pairs; any change to the label rules must be made before training and documented, never after seeing
  results.
- *Training pools*: 20,000 presence and 20,000 counting pairs over the 8,713 cached training images, sampled uniformly
  over eligible categories.
- *Connector calibration*: fc2 is scaled once at initialization by (mean training-text embedding norm) /
  (mean initial visual-token norm on 512 training images); rule and scale are saved.
- *Optimizer per stage*: a fresh AdamW at each stage boundary (the parameter set and peak LRs change); weight decay
  excludes biases and 1-D tensors as in Task 1.
- *Normalization*: the official VQA evaluation normalization is applied to all references and predictions.

**Task 3**
- *Answer equivalence (MATH)*: math-verify is run with its internal timeouts disabled (they rely on signals or
  subprocesses that fail on Windows) inside a worker pool with a 120 s per-question timeout. One question
  (`test:4928`) fell back to exact-string grouping. Groups are formed against group representatives in rollout-ID
  order without gold answers; each group is graded with `verify(gold, representative)`.
- *MathVista rules*: multiple choice -> option letters; integers / floats parsed numerically (units, `\text{}`,
  degree and percent signs removed; floats rounded to the question's precision); lists canonicalized.
- *Random orders*: `default_rng([6304, 1])` per dataset over questions in sorted-ID order; the split permutation uses
  `default_rng(6304)` as specified.
- *Live seeds*: SHA-256 of `6304|sample_id|strategy|k`, one independent seed per call.

## 6. Attribution

- VQA answer normalization in `task2_vlm/vqa_norm.py` is ported from the official VQA evaluation code,
  https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py (contractions,
  number map, punctuation and article handling copied; wrapped into functions).
- Benchmarks use EleutherAI lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness) stock tasks.
- MATH answer equivalence uses Hugging Face Math-Verify (https://github.com/huggingface/Math-Verify).
- Qualitative prompts: `first_plot_questions.yaml` from https://github.com/emergent-misalignment/emergent-misalignment.
- Code was written with LLM coding assistance (Claude), as permitted by the manual's coding-assistance policy.
