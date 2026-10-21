---
title: Training a model
description: The training loop, the optimisers, and planning for a small card.
section: Guides
order: 4
---

# Training a model

## The training loop

A training loop is a loop. Nothing special about it.

```milo
let model = linear(3, 2)
let optimizer = adamw(model.parameters, 0.05)

let inputs = tensor([[1.0, 2.0, 3.0]])
let targets = tensor([[0.5, -0.25]])

for step in range(60):
    let prediction = model(inputs)
    let loss = mse_loss(prediction, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print "step " + str(step) + ": loss " + str(loss)
```

```
step 0: loss 4.2855
step 20: loss 0.212994
step 40: loss 0.00637241
the final loss:
0.00153914
```

Three steps, and each of them is one line:

| | |
|---|---|
| `optimizer.zero_grad()` | forget the gradients from the last step |
| `loss.backward()` | work out the gradient of the loss with respect to every trainable tensor |
| `optimizer.step()` | update the parameters |

## The parts

**Layers** are callable values with named parameters.

```milo
let model = linear(3, 2)
let norm = layer_norm(4)

let output = model(inputs)
print model.parameters["weight"]
```

**Losses** give a single number.

```milo
let loss = mse_loss(prediction, targets)
let loss = cross_entropy(logits, targets)
```

**Optimisers** hold the parameters they update.

```milo
let optimizer = sgd(model.parameters, 0.01)
let optimizer = adam(model.parameters, 0.001)
let optimizer = adamw(model.parameters, 0.001)
```

`adamw` is `adam` with the weight decay taken out of the gradient, which
is usually what you want.

## Gradients

Every operation on a trainable tensor is recorded, so that the gradients can
be worked out afterwards. A tensor asks for a gradient by being made
trainable:

```milo
let w = trainable(tensor([1.0, 2.0, 3.0]))
```

Layers do this for you. The arithmetic operators work on tensors too, and
go through the same tape:

```milo
let doubled = w * 2.0
let loss = sum(doubled)
loss.backward()
print w.grad
```

## Planning for a small card

Before you start a long run, ask what it will need.

```bash
milo plan model.safetensors --batch-size 4 --sequence 2048 --memory 12 --lora
```

```
  weights             13.0 GiB
  gradients          112.0 MiB
  optimizer state    224.0 MiB
  activations         22.0 GiB
  attention scores    32.0 GiB
  total               67.5 GiB

  it does not fit: it needs 12429607302 bytes of 12884901888

  What would make it fit, in the order to try it:
    1. batch_size: 4 -> 2
         smaller batches use less memory; raise the gradient accumulation to
         keep the effective batch the same
    2. gradient_checkpointing: off -> on
         the activations are recomputed instead of kept, which costs about
         a third of the speed
    3. sequence_length: 2048 -> 1024
         longer examples are truncated, which loses the end of the long ones
    4. quantization: none -> int8
         the weights are stored more compactly, which costs a little accuracy

  Nothing has been changed. Apply these yourself if you want them.
```

**The planner never changes anything.** It reports what would help and in
what order; applying it is your decision. The same answer is available from
a program:

```milo
let plan = memory_plan(model, batch_size = 4, sequence = 2048,
                       device_memory = 12, lora = true, lora_rank = 16)

print plan["fits"]
print plan["total"]
for suggestion in plan["suggestions"]:
    print suggestion["knob"] + ": " + suggestion["from"] + " -> " + suggestion["to"]
```

## Being honest about it

Milo's tooling makes **fine-tuning** feasible on modest hardware. It does
not make pretraining a large model from scratch cheap: a from-scratch
one-billion-parameter pretrain still needs a cluster. The planner will tell
you so, rather than pretending otherwise.
