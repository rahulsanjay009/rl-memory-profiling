# RL Post-Training GPU Memory Characterization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahulsanjay009/rl-memory-profiling/blob/main/notebooks/01_trl_baseline_profile.ipynb)

**Scope note:** this is a measurement study. It instruments and characterizes how GPU memory is used across the rollout (generation) and update (training) phases of RLHF-style post-training, and reports time-resolved traces and breakdowns.

## Motivation

During RL post-training of LLMs (PPO/GRPO-style), each step alternates between two phases with very different memory profiles:

- **Rollout**: the policy generates responses; GPU memory holds model weights plus a growing KV cache.
- **Update**: the optimizer applies a policy update; GPU memory holds weights, gradients, and optimizer state (e.g. Adam's first/second moment estimates).

Both phases can be memory-dominant, just at different times within the same step. Existing open-source RL post-training systems (veRL/HybridFlow, LlamaRL) and memory-offloading systems (ZeRO-Offload, ZenFlow) are well documented in terms of throughput and peak memory, but there's less public, reproducible data on the **time-resolved shape** of memory usage within a step - which tensors are idle, for how long, in which phase. That's what this repo measures.

## Related work

1. Sheng, G. et al. **"HybridFlow: A Flexible and Efficient RLHF Framework."** EuroSys 2025. [arXiv:2409.19256](https://arxiv.org/abs/2409.19256). The veRL framework, used here as the primary profiling target.
2. Wang, X. et al. (Meta) **"LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Training."** [arXiv:2505.24034](https://arxiv.org/abs/2505.24034). Async rollout/update overlap at scale, motivates why phase boundaries matter.
3. Ren, J., Rajbhandari, S., et al. **"ZeRO-Offload: Democratizing Billion-Scale Model Training."** [arXiv:2101.06840](https://arxiv.org/abs/2101.06840). The baseline host-DRAM offload approach Experiment 5 reproduces a measurement from.
4. Lan, T. et al. **"ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates."** [arXiv:2505.12242](https://arxiv.org/abs/2505.12242). Shows the cost of naive CPU offloading (PCIe-bound stalls), which is the motivation for finer-grained tiering.
5. **"Rollplex: Cross-Phase GPU Spatial Sharing for Vision Language Model Post-Training."** [arXiv:2608.14498](https://arxiv.org/abs/2608.14498). Phase-aware HBM residency management across rollout/training phases, close to this repo's phase-tagging approach. See also RollMux, [arXiv:2512.11306](https://arxiv.org/abs/2512.11306).
6. Feng, Y. et al. **"Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation."** [arXiv:2407.11550](https://arxiv.org/abs/2407.11550). *Substituted for "AdaptiveKV," which didn't resolve to a verifiable title. Swap in the exact paper if you had a different one in mind.*
7. Wang, X., Liu, J., Wu, J., Yang, S., Ren, J., Shankar, B., Li, D. **"Exploring and Evaluating Real-world CXL: Use Cases and System Adoption."** IPDPS 2025. [arXiv:2405.14209](https://arxiv.org/abs/2405.14209). Dong Li's group's measured characterization of CXL memory for HPC/LLM workloads on real hardware. Gives directly relevant baseline numbers (PCIe-bottlenecked GPU-CXL tensor copy) for any future offload-latency comparison.

## Setup

**Model:** Qwen2.5-0.5B-Instruct. Small enough for single-GPU iteration, and it's the model veRL's own quickstart examples use, so results stay comparable to the framework's documented numbers.

**Two tracks, in order of increasing fidelity to the target systems:**

- **Track A, TRL `GRPOTrainer`:** pure `transformers` + `trl`, no `vllm`/`ray` dependency. TRL's classic `PPOTrainer` has moved to an experimental namespace with a much heavier setup (separate reward + value models). `GRPOTrainer` is TRL's current primary on-policy trainer and only needs a reward function and a prompt dataset. Runs on a single consumer GPU (16GB+) or Colab's free T4. Use this to get the harness working end-to-end cheaply before touching veRL.
- **Track B, veRL (HybridFlow):** run veRL's official single-GPU PPO/GRPO quickstart (Qwen2.5-0.5B on GSM8K). This is the actual framework named in the related work, so its memory behavior is the one worth publishing. Needs ~24GB VRAM with small batch sizes. If local hardware doesn't fit, a single rented A100/H100 (RunPod, Lambda, Vast.ai, a few dollars for a few hours) is enough.

## Instrumentation and metrics

- **Coarse timeline:** background thread polling `torch.cuda.memory_allocated()` / `memory_reserved()` every 50-100ms, written to CSV with wall-clock timestamps.
- **Phase tagging:** wrap the known call sites with a context manager that logs phase-start/phase-end timestamps, so the memory CSV can be labeled rollout vs. update after the fact. `actor.generate_sequences(...)` for rollout and `actor.update_policy(...)` for update in veRL; `GRPOTrainer._generate_and_score_completions(...)` for rollout and `GRPOTrainer.training_step(...)` for update in TRL.
- **Category breakdown:** direct introspection of live tensors (`compute_category_breakdown` in `scripts/memory_poller.py`), summing parameter, gradient, and optimizer-state tensor bytes at a given checkpoint. More robust for a public repo than parsing the CUDA caching allocator's internal snapshot format, which needs heuristics to attribute blocks to categories. For a deeper dive including allocator fragmentation, record a snapshot with `torch.cuda.memory._record_memory_history(enabled=True)` + `torch.cuda.memory._dump_snapshot()` and inspect it with PyTorch's [memory_viz](https://pytorch.org/memory_viz) tool.
- **Independent cross-check:** `nvidia-smi --query-gpu=memory.used --format=csv -l 1` (or `nvidia-smi dmon`) logged in parallel, since it measures from outside the PyTorch allocator and catches anything the allocator hides (fragmentation, CUDA context overhead, vLLM's own memory pool in Track B).
- **Optional, for a polished figure:** `nsys profile` with `torch.cuda.nvtx.range_push("rollout")` / `range_push("update")` markers around the same call sites, for an annotated kernel-level timeline.

## Experiments

1. **Baseline phase memory profile.** Run N steps of Track B, collect the full CSV + phase labels, and plot memory vs. time colored by phase. This is the headline artifact. It should visually show the handoff between KV-cache growth during rollout and optimizer/gradient growth during update.
2. **Memory category breakdown.** Take memory snapshots at three points, mid-rollout, start-of-update, peak-of-update, and report each category's share of total footprint at each point. Shows which category dominates in which phase.
3. **Idle-time characterization.** For each category from (2), compute how long it sits allocated but untouched between its last access in one phase and its next access in the following phase, as a fraction of total step time. For example: "optimizer state sits idle for X% of the step."
4. **Workload sweep (roadmap, not run here).** Repeat (1)-(3) while varying rollout batch size and max generation length, to see whether the idle-time fraction from (3) is sensitive to workload shape. Expected payoff: a workload-dependent curve showing that any downstream memory-management decision needs to be workload-aware, not built around one fixed threshold.
5. **Offload cost baseline (roadmap, not run here).** Enable DeepSpeed's ZeRO-Offload on optimizer state only (or a manual `.to('cpu', non_blocking=True)` round-trip) and measure the added per-step latency, reproducing ZeRO-Offload/ZenFlow's measurement style. Expected payoff: a concrete per-step latency cost for host-DRAM round-trips, the reference baseline any future memory-tier comparison would need.
6. **(Stretch, roadmap) Annotated Nsight timeline.** A clean `nsys` screenshot with rollout/update regions marked. Expected payoff: a kernel-level, publication-quality figure of the phase structure, finer-grained than the Python-level polling used in (1)-(3).

## Repository structure

```
rl-memory-profiling/
  README.md
  requirements.txt
  notebooks/
    01_trl_baseline_profile.ipynb   # runs live on Colab (Track A), plots rendered inline
    02_verl_phase_analysis.ipynb    # analyzes traces from a veRL run (Track B)
  scripts/
    memory_poller.py        # background CUDA memory sampler + phase tagging + category breakdown
    plot_memory_timeline.py # CSV -> annotated plot (standalone, non-notebook use)
  results/
    <date>_<track>_<config>.csv
    plots/
```

## How to run

**Track A (TRL):** open `notebooks/01_trl_baseline_profile.ipynb` directly in Colab (badge above). It installs dependencies, runs GRPO on Qwen2.5-0.5B-Instruct, and renders the phase-memory plot, category breakdown, and idle-time numbers inline. Needs a GPU runtime (Runtime -> Change runtime type -> GPU); the free T4 is enough.

**Track B (veRL):** veRL's Ray/vLLM orchestration doesn't run cleanly inside a hosted notebook, so run it on a GPU box (local or rented, RunPod/Lambda/Vast.ai) via the script, then analyze the resulting trace in a notebook:

```bash
pip install -r requirements.txt

# on the GPU box, after installing veRL per its own docs and wiring the
# phase markers described in scripts/memory_poller.py's run_verl_track:
python scripts/memory_poller.py --track verl --config examples/ppo_trainer/qwen2.5_0.5b_gsm8k.yaml --steps 20 --out results/verl_trace.csv
```

Then copy `results/verl_trace.csv` into the repo and open `notebooks/02_verl_phase_analysis.ipynb` to reproduce the same plots/analysis as Track A. `scripts/plot_memory_timeline.py` is also there for a quick non-notebook plot from any CSV trace.

## Citation

If useful, cite this repo and the papers above.
