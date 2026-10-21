#!/usr/bin/env python3
"""Create the safetensors fixture used by the test suite.

The format is simple enough to write by hand, which is also a good check
that the reader in `src/safetensors` agrees with the specification.

    python tools/make_fixture.py tests/scripts/fixture.safetensors
"""

import json
import struct
import sys


def f32(values):
    return b"".join(struct.pack("<f", float(v)) for v in values)


def i64(values):
    return b"".join(struct.pack("<q", int(v)) for v in values)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/scripts/fixture.safetensors"

    tensors = [
        ("layer.weight", "F32", [2, 3], f32([1, 2, 3, 4, 5, 6])),
        ("layer.bias", "F32", [3], f32([0.5, -0.5, 0.25])),
        ("step.count", "I64", [], i64([42])),
    ]

    metadata = {"format": "pt", "note": "Milo test fixture"}

    # The data is laid out in the order the tensors appear, and the offsets
    # are relative to the start of the data buffer.
    header = {"__metadata__": metadata}
    offset = 0
    blobs = []
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for blob in blobs:
            out.write(blob)

    print(f"wrote {path} ({offset + len(header_bytes) + 8} bytes)")


if __name__ == "__main__":
    main()
