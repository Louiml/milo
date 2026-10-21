---
title: The memory planner
description: How Milo works out whether a training run will fit, and what to do about it.
section: Reference
order: 13
---
# Milo — Memory planner

Status: designed in Phase 0, implemented in Phase 5.

## 1. Purpose

Before a training run starts, Milo answers two questions:

1. **Will this fit?** How much VRAM/RAM does this configuration need?
2. **If not, what should I change?** Produce a concrete, ordered list of
   adjustments that make it fit.

The planner is not a suggestion engine. It returns a *plan*: an ordered list
of `(knob, from, to, reason)` that the runtime applies automatically, unless
the user asked for `--no-auto-adjust`.

## 2. The model

### 2.1 Parameters

For a model with parameters `P` (count) and a given configuration, training
memory is dominated by these terms:

```
weights        = P * sizeof(dtype)                    # frozen + trainable
gradients      = P_trainable * sizeof(dtype)
optimizer      = P_trainable * 2 * sizeof(f32)         # Adam m and v
activations    = sum over layers of act_bytes(layer, batch, seq)
scratch        = workspace for the largest single op
```

`act_bytes` for a transformer block at batch `b` and sequence `s` with hidden
`h` is roughly:

```
attention_scores = b * heads * s * s * sizeof(act_dtype)
mlp_hidden       = b * s * (4h)   * sizeof(act_dtype)
```

Note the `s * s` term: attention memory grows quadratically with sequence
length. This is the single most important fact for anyone fine-tuning on a
laptop, and the planner surfaces it explicitly.

### 2.2 Quantization adjustments

* QLoRA (NF4 base weights): base weights are 4 bits + scales ≈ 0.5 bytes/param
  instead of 2 (bf16) or 4 (f32). Scales add ~0.03 bytes/param.
* Trainable LoRA parameters are a tiny fraction of `P`
  (`rank * (d_in + d_out)` per adapted matrix), so gradients and optimizer
  state are small.
* Activations keep the compute dtype (bf16/fp16); the base weights are
  dequantized on the fly.

### 2.3 Offloading

Layers that are not being computed can live in host RAM. The planner reports
how many layers must be offloaded to fit, and the runtime streams them.

## 3. Algorithm

```
plan(config, device):
    estimate = estimate_memory(config)
    if estimate.total <= device.budget:
        return plan_ok(estimate)

    steps = []
    cfg = config.copy()

    for knob in ORDER_OF_PRECEDENCE:
        if cfg[knob] == knob.minimum: continue
        candidate = cfg.with(knob, knob.minimum)
        if estimate(candidate).total <= device.budget:
            steps.append(knob, cfg[knob], knob.minimum, knob.reason)
            cfg = candidate
            break
        else:
            cfg = candidate   # keep going down, it only gets smaller

    if estimate(cfg).total > device.budget:
        return plan_impossible(estimate(cfg), device)

    return plan_with_adjustments(estimate(cfg), steps)
```

`ORDER_OF_PRECEDENCE` — cheapest quality loss first:

1. `batch_size` 1 → 2 → 4 (quality loss: noisy gradients)
2. `gradient_checkpointing` on (quality: ~30% slower, memory: −40% activations)
3. `max_length` truncate (quality: loses long examples)
4. `quantization` int8 → nf4 (quality: small accuracy drop)
5. `precision` f32 → bf16 → f16 (quality: usually none)
6. `optimizer` adamw → adam → sgd (quality: worse convergence)
7. `offload_layers` (quality: slower, memory: moves to RAM)
8. `trainable_layers` subset / freeze embeddings (quality: task-dependent)

Gradient accumulation is the counter-move: if batch size must drop, raise
`gradient_accumulation` to keep the effective batch the same. The planner
offers this explicitly, because dropping the effective batch size silently is
a common way to get a worse model and not know why.

## 4. Output

```
milo plan --model llama-3-8b.safetensors --data chats.csv --batch-size 4

  device            : cuda:0 (12.0 GiB usable of 12.0 GiB)
  base weights      : 8.03 GB   (bf16, 8.03 B params)
  trainable params  : 10.5 M    (LoRA r=16, q,k,v,o,gate,up,down)
  gradients         : 42 MB
  optimizer state   : 84 MB     (adamw, f32 m+v)
  activations       : 1.94 GB   (batch 4, seq 2048, 32 layers)
  attention scores  : 1.61 GB   (32 heads, seq 2048)
  scratch           : 128 MB
  -----------------------------------------------------------
  total             : 11.2 GB
  budget            : 12.0 GB
  result            : fits

  with the requested settings: fits

  with --batch-size 8:
    total           : 16.8 GB   (over by 4.8 GB)
    adjustments     :
      1. batch size          8 -> 4        (fits the device)
      2. gradient accumulation 1 -> 2     (keeps the effective batch at 8)
```

## 5. Accuracy of the estimate

The estimate is a model, not a measurement. Milo reports the estimate and, in
`--verbose` mode, compares it against the **measured peak allocation** after a
real run, and prints the error. The docs will publish a table of estimate vs.
measured across models, so users know how much to trust it.

Target: within 15% of measured peak on the supported backends. If a model
family consistently misses that, the coefficients get fixed.

## 6. Honesty requirements

* The planner never claims a run will fit when it only fits in theory. If the
  estimate says "fits" but the machine has 4 GB of other things running, the
  number shown is the *usable* budget, not the *total* device memory.
* If a configuration genuinely cannot fit (e.g. full fine-tuning a 70B model
  on an 8 GB card), the planner says so plainly and explains what would be
  required — it does not pretend a magic combination will work.
* Full fine-tuning of a large model is not "impossible on a laptop"; it is
  "requires ~X GB, which you do not have". The planner prints the requirement.
