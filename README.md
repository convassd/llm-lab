# llm-lab
.venv\Scripts\Activate.ps1

This is my small-data LLM lab. Weekly cadence: **1 commit + 1 plot + 1 note**.
Goal: hybrid SNN-Transformer prototypes by Month 14.

## Phase 1 — Core chops & muscle memory (Weeks 1–12)

* W01 (8/10/2025) Minimal bigram LM trains end-to-end; save loss.png + sample.txt.
* W02 (8/17/2025) Sampling CLI, config file, deterministic seeds; refactor data loader.
* W03 (8/24/2025) Toy BPE tokenizer; encode/decode tests; tiny benchmark.
* W04 (8/31/2025) Train tiny char-GPT (1–2 heads); show samples vs steps.
* W05 (9/7/2025) Harden loop: grad clip, cosine LR, early stop; tidy logs.
* W06 (9/14/2025) 50-line MNIST CNN (fluency drill); learning-curve plot.
* W07 (9/21/2025) Reusable experiment template (configs, logger, run dirs).
* W08 (9/28/2025) Reproduce nano-GPT baseline (super small); record perplexity.
* W09 (10/5/2025) Implement full TransformerBlock (MH-Attn + MLP + LayerNorm).
* W10 (10/12/2025) Train tiny GPT on TinyShakespeare + small zh subset; compare.
* W11 (10/19/2025) Efficiency: torch.compile, grad checkpointing; stability notes.
* W12 (10/26/2025) Micro-eval harness (perplexity + toy tasks); tag minigpt-from-zero v0.1.

## Phase 2 — Small LLMs that work (Weeks 13–24)

* W13 (11/2/2025) QLoRA pipeline (≤3B, 4-bit); smoke test.
* W14 (11/9/2025) Curate tiny SFT set; first finetune; losses & samples.
* W15 (11/16/2025) Micro-bench evals (math toy, short reasoning); exact-match metrics.
* W16 (11/23/2025) Inference packaging: CLI sampler + minimal FastAPI endpoint.
* W17 (11/30/2025) Data curation EN/zh; formatting, dedupe; loader utils.
* W18 (12/7/2025) Packing/grad accum/LR sweep; memory/throughput table (RTX 3060).
* W19 (12/14/2025) Robustness: NaN/grad-norm checks; checkpoint/restart; ablations doc.
* W20 (12/21/2025) Publish qlora-finetune-kit v0.1; README + quickstart.
* W21 (12/28/2025) Expand micro-evals; prompt templates; variability report.
* W22 (1/4/2026) Context length & sampling (temp/top-k/p) study; quality vs speed.
* W23 (1/11/2026) KV cache + generation latency; tokens/s plot.
* W24 (1/18/2026) Ship llm-api-starter (optionally Docker); Phase summary note.

## Phase 3 — SNN basics → Hybrid SNN-Transformer (Weeks 25–60)

* W25 (1/25/2026) LIF neuron + surrogate grads; unit tests.
* W26 (2/1/2026) STDP demo on toy data; visualize weight changes.
* W27 (2/8/2026) SpikingJelly or Norse trial; train tiny spiking net on sequences.
* W28 (2/15/2026) Text-to-spike v0 (rate coding); compare to raw tokens on toy task.
* W29 (2/22/2026) Temporal/latency coding variant; A/B against rate coding.
* W30 (3/1/2026) Package text2spike module; micro-benchmarks + docs.
* W31 (3/8/2026) **Hybrid #1:** Spiking encoder → tiny Transformer; end-to-end.
* W32 (3/15/2026) Baselines (no SNN front-end); sample-efficiency curves.
* W33 (3/22/2026) Sweep: spike thresholds, tau, embed size; ablation table.
* W34 (3/29/2026) Sparsity metrics (activity %, proxy energy); correlate with quality.
* W35 (4/5/2026) Working-memory toy tasks (list recall, distract-then-recall); scores.
* W36 (4/12/2026) Tech-report draft (Intro/Related/Methods); figures WIP.
* W37 (4/19/2026) **Hybrid #2:** Spiking gate/controller for attention heads/layers.
* W38 (4/26/2026) Train gated vs non-gated; tokens-to-target-perplexity comparison.
* W39 (5/3/2026) Noise/perturbation robustness tests; degradation curves.
* W40 (5/10/2026) Clean & release spikeformer-lite v0.1; README + examples.
* W41 (5/17/2026) Add small Chinese corpora; cross-lingual sample-efficiency check.
* W42 (5/24/2026) Visualization: spike rasters + attention maps; interactive nb.
* W43 (5/31/2026) Low-data regimes (1k/5k/10k tokens); learning curves + error bars.
* W44 (6/7/2026) Repro harness: seeds, fixed splits, `make repro` script.
* W45 (6/14/2026) Package quality: type hints, docstrings, tests; optional CI.
* W46 (6/21/2026) Minimal web demo (text → spikes/gates → continuation).
* W47 (6/28/2026) Dry-run talk (slides); peer feedback; action items.
* W48 (7/5/2026) Polish pass; tag spikeformer-lite v0.2.
* W49 (7/12/2026) Hybrid #2 final: working-memory loop or SNN-gated attention v2.
* W50 (7/19/2026) Train & tune Hybrid #2; stabilize; learning curves.
* W51 (7/26/2026) Head-to-head: Hybrid #1 vs #2 vs baseline; unified plots.
* W52 (8/2/2026) Buffer/Cleanup week — refactor, flaky tests, doc gaps.
* W53 (8/9/2026) Curriculum learning on tiny datasets; convergence vs curriculum.
* W54 (8/16/2026) Encoder-only vs decoder-only ablation; results table.
* W55 (8/23/2026) Memory-length stress test; (tiny) scaling-law plot.
* W56 (8/30/2026) Paper draft v0 (Results/Discussion); bibliography tidy.
* W57 (9/6/2026) Data/model cards; licenses; repo housekeeping.
* W58 (9/13/2026) Colab/CPU demo; “try it in 5 minutes” instructions.
* W59 (9/20/2026) Blog post outline + figures; small-data messaging.
* W60 (9/27/2026) External feedback round; incorporate edits; plan final runs.

## Phase 4 — Polish, position, publish (Weeks 61–78)

* W61 (10/4/2026) Capstone reruns (best hybrid); largest ablation set.
* W62 (10/11/2026) Multi-seed runs; error bars & stats; final plots.
* W63 (10/18/2026) Package to pip (installable); tests ≥80% on critical path.
* W64 (10/25/2026) Demo hardening; edge cases; latency budget; UX pass.
* W65 (11/1/2026) Final 6–8 page report text; figure polish; captions.
* W66 (11/8/2026) Upload arXiv/tech report; project page live.
* W67 (11/15/2026) CV refresh + portfolio links; short hiring blurb.
* W68 (11/22/2026) Talk deck + demo video; rehearse.
* W69 (11/29/2026) Community share; gather testimonials.
* W70 (12/6/2026) Buffer/maintenance week; fix issues, triage bugs.
* W71 (12/13/2026) Domain-adaptation mini-study (new tiny corpus); results.
* W72 (12/20/2026) Post-feedback update; tag v1.0.
* W73 (12/27/2026) Docs final: API refs, diagrams, examples.
* W74 (1/3/2027) Mock interviews / whiteboard practice (LLM systems).
* W75 (1/10/2027) Application sprint; track outreach; follow-ups.
* W76 (1/17/2027) Reference letters / endorsements; finalize materials.
* W77 (1/24/2027) Retrospective: what worked, what didn’t; next-phase ideas.
* W78 (1/31/2027) Celebrate; archive; set Q4 goals.
