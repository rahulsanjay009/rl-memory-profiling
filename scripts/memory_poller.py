"""
GPU memory profiling harness for RL post-training phase characterization.

The core piece (MemoryPoller) is self-contained and correct for any PyTorch
workload. The TRL integration below is a worked example; the veRL integration
is left as a commented skeleton since its internal call sites shift between
releases -- adjust the two `with poller.phase(...)` insertion points to match
whichever actor/trainer methods your installed veRL version calls for
generation and for the policy update.
"""

import argparse
import csv
import threading
import time
from contextlib import contextmanager

import torch


class MemoryPoller:
    """Background sampler of CUDA memory, labeled with the active phase."""

    def __init__(self, interval_s: float = 0.05, device: int = 0):
        self.interval_s = interval_s
        self.device = device
        self._phase = "idle"
        self._phase_lock = threading.Lock()
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    @contextmanager
    def phase(self, name: str):
        with self._phase_lock:
            previous = self._phase
            self._phase = name
        try:
            yield
        finally:
            with self._phase_lock:
                self._phase = previous

    def _poll_loop(self):
        while not self._stop.is_set():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            with self._phase_lock:
                phase = self._phase
            self._samples.append((time.time(), phase, allocated, reserved))
            time.sleep(self.interval_s)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def to_csv(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "phase", "allocated_bytes", "reserved_bytes"])
            writer.writerows(self._samples)

    def to_dataframe(self):
        import pandas as pd

        df = pd.DataFrame(self._samples, columns=["timestamp", "phase", "allocated_bytes", "reserved_bytes"])
        df["t"] = df["timestamp"] - df["timestamp"].min()
        return df


def compute_category_breakdown(model, optimizer=None):
    """
    Direct introspection of live parameter/gradient/optimizer-state tensors,
    in bytes. More robust for a public repo than parsing the CUDA caching
    allocator's internal snapshot format (torch.cuda.memory._dump_snapshot),
    which requires heuristics to attribute blocks to categories. For a deeper
    dive including allocator fragmentation, record a snapshot with
    `torch.cuda.memory._record_memory_history` and inspect it at
    https://pytorch.org/memory_viz.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )

    optimizer_state_bytes = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    optimizer_state_bytes += value.numel() * value.element_size()

    return {
        "parameters": param_bytes,
        "gradients": grad_bytes,
        "optimizer_state": optimizer_state_bytes,
    }


def run_trl_track(model_name: str, steps: int, poller: MemoryPoller):
    """
    Minimal PPO loop against TRL, with rollout/update phases tagged.

    TRL's trainer API has changed across releases (PPOTrainer vs PPOv2Trainer
    vs the unified trainer in newer versions) -- check `pip show trl` and
    TRL's docs for your installed version, and adjust the two call sites
    below (the `.generate(...)` call and the `.step(...)`/update call)
    accordingly. The phase-tagging pattern itself does not change.
    """
    from transformers import AutoTokenizer
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).cuda()
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).cuda()

    config = PPOConfig(batch_size=4, mini_batch_size=2)
    trainer = PPOTrainer(config, model, ref_model, tokenizer)

    prompt = "Explain why the sky is blue."
    query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    for step in range(steps):
        with poller.phase("rollout"):
            response_tensor = trainer.generate(query_tensor, max_new_tokens=64)

        reward = torch.tensor([1.0]).cuda()  # placeholder reward signal

        with poller.phase("update"):
            trainer.step([query_tensor[0]], [response_tensor[0]], [reward])

        print(f"step {step} done")


def run_verl_track(config_path: str, steps: int, poller: MemoryPoller):
    """
    Skeleton for veRL (HybridFlow) integration.

    This is intentionally not a runnable end-to-end script: wire
    `with poller.phase("rollout")` around whichever method your veRL
    version uses to generate rollouts (e.g. an actor's generate_sequences
    call), and `with poller.phase("update")` around the policy update
    call (e.g. update_policy / update_actor), inside veRL's own trainer
    loop. Follow veRL's own install + quickstart docs
    (https://github.com/volcengine/verl) to get the base PPO/GRPO run
    working first, then add the two phase markers at those call sites.

    To feed notebooks/02_verl_phase_analysis.ipynb, also call
    `compute_category_breakdown(actor_module, actor_optimizer)` right before
    and right after the update call, same as the TRL notebook does, and
    write those rows to results/verl_breakdown.csv (step, checkpoint,
    parameters, gradients, optimizer_state).
    """
    raise NotImplementedError(
        "Install veRL per its own docs, then add poller.phase(...) and "
        "compute_category_breakdown(...) calls around its rollout and "
        "update call sites as described above."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["trl", "verl"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--config", default=None, help="veRL config path (track=verl)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--out", default="results/trace.csv")
    args = parser.parse_args()

    poller = MemoryPoller(interval_s=args.interval)
    poller.start()
    try:
        if args.track == "trl":
            run_trl_track(args.model, args.steps, poller)
        else:
            run_verl_track(args.config, args.steps, poller)
    finally:
        poller.stop()
        poller.to_csv(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
