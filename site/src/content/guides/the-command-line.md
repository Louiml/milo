---
title: The command line
description: Everything `milo` can do, and the package manager beside it.
section: Guides
order: 5
---

# The command line

## `milo`

```
milo run <file.milo>     run a program
milo check <file.milo>   parse a program and report problems, without running it
milo inspect <file>      report on a safetensors file
milo plan <model>        work out what a training run would need
milo repl                start the interactive prompt
milo version             print version information
milo help                show this message
```

Options:

| | |
|---|---|
| `-e, --eval <source>` | run a snippet given on the command line |
| `--vm` | run with the bytecode machine instead of the evaluator |

## `milo inspect`

A report on a safetensors file: the sizes, a table of the tensors, a
breakdown by dtype and by shape, and the metadata.

```bash
milo inspect model.safetensors
```

## `milo plan`

What a training run would need, and what could be changed to make it fit.

```bash
milo plan model.safetensors \
    --batch-size 4 \
    --sequence 2048 \
    --memory 12 \
    --lora --lora-rank 16 \
    --quantization nf4
```

The options are:

| | |
|---|---|
| `--batch-size N` | how many examples are trained at once |
| `--sequence N` | how long each example is |
| `--memory N` | how much memory the device has, in gibibytes |
| `--hidden N` | the model's hidden size |
| `--layers N` | how many layers the model has |
| `--heads N` | how many attention heads the model has |
| `--parameters N` | how many parameters the model has |
| `--quantization none\|int8\|nf4` | how the weights are stored |
| `--lora` | assume low-rank fine-tuning |
| `--lora-rank N` | the rank of the adapters |
| `--gradient-checkpointing` | assume the activations are recomputed |
| `--gradient-accumulation N` | how many batches are accumulated |
| `--offload-optimizer` | assume the optimiser's state is on the host |
| `--device NAME` | the device's name, for the report |

The model's hidden size, layer count and head count are guessed from the
file, and every guess can be overridden.

## `mpm`

The package manager. It is written in Rust, it talks to the runtime only
through the C ABI, and it has no Cargo dependencies of its own.

```
mpm init [name]        create a milo.toml manifest in the current directory
mpm add <spec>         add a dependency to the manifest
mpm remove <name>      remove a dependency from the manifest
mpm install            resolve and install the dependency tree
mpm build              build the current package
mpm publish            publish the current package to the registry
mpm update [name]      update dependencies to the latest allowed versions
mpm search <query>     search the registry index
mpm runtime            report the runtime this build is linked against
```

```bash
mpm init my-project
mpm add "milo-numpy@^0.2"
mpm install
```

## The C ABI

Other programs can use Milo through `include/milo/milo.h`. It is plain C,
exception-free, and versioned.

```c
#include <milo/milo.h>

milo_runtime* rt = milo_runtime_create();
milo_value*   out = NULL;

milo_status status = milo_runtime_run(rt, "let x = 21\nx * 2\n", 0, "demo.milo", &out);
if (status != MILO_OK) {
  char buffer[512];
  milo_last_error(buffer, sizeof(buffer));
  fprintf(stderr, "%s", buffer);
}

milo_value_release(out);
milo_runtime_destroy(rt);
```
