---
title: Packages with mpm
description: The package manager — manifests, lockfiles, a local registry, and working offline.
section: Guides
order: 7
---

# Packages with `mpm`

`mpm` is Milo's package manager. It is written in Rust, it has no Cargo
dependencies at all, and it talks to the Milo runtime only through the
stable C ABI.

## A package

A package is a directory with a `milo.toml` manifest in it.

```console
$ mpm init my-project
created milo.toml
next: add dependencies with `mpm add <name>`
```

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2026"
license = "MIT"

[dependencies]
milo-numpy = "^0.1"
```

## Dependencies

```console
$ mpm add "milo-numpy@^0.2"
added milo-numpy ^0.2.0

$ mpm remove milo-numpy
removed milo-numpy
```

The versions are ordinary semantic versions, and the requirements are the
usual ones: `1.2.3` exactly, `^1.2.3` compatible, `~1.2.3`
approximately, `>=1.2.3` at least, `*` anything.

## Installing

```console
$ mpm install
installed milo-numpy 0.1.0
wrote milo.lock
```

`mpm install` resolves the tree, downloads what is missing, checks every
archive against the SHA-256 in the registry index, and unpacks it into
`milo_packages/`.

The lockfile records what was installed, not what was asked for:

```json
{
  "lockfile": 1,
  "packages": [
    {
      "name": "milo-numpy",
      "sha256": "fa176a6e013b3d814f5ed0eafe68c5da2ff1a86274c53cf8f31ad79b2e433ee5",
      "source": "milo-numpy-0.1.0.mpkg",
      "version": "0.1.0"
    }
  ]
}
```

Install again and the same bytes come back.

## Offline

Every download is kept in a cache, and `--offline` turns the network off
for good.

```console
$ mpm install --offline
installed milo-numpy 0.1.0
```

If something is not in the cache, `mpm` says so and stops, rather than
reaching for the network behind your back.

## A registry

A registry is a directory with an `index.json` in it, or an HTTP server
that serves one. The index is a static file: publishing adds a release
to it.

```console
$ mpm publish --registry .milo/registry
published my-project 0.1.0 to .milo/registry
sha256 35ea4743fd7a81942c68ca3cd1a61855d181f76970e8c86dae8e0ce627057a96
```

```console
$ mpm search numpy --registry .milo/registry
milo-numpy (0.1.0, 0.2.0)
```

The registry to use can also be set in the environment.

```console
$ export MILO_REGISTRY=https://registry.example.invalid
```

## Building

```console
$ mpm build
built target/my-project-0.1.0.mpkg
sha256 35ea4743fd7a81942c68ca3cd1a61855d181f76970e8c86dae8e0ce627057a96
```

A package is its own manifest, its own source, and its own hash.

## The whole story

The end-to-end test in `tools/mpm/tests/cli.rs` does all of this: a
package is written, built, published, found by another project,
installed, and installed again with the network off.

```console
$ cargo test -p mpm
test result: ok. 38 passed; 0 failed
```
