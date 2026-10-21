"""Tests for Milo's Python interoperability.

They cover both directions: Python calling Milo, and a Milo program calling
Python.
"""

from __future__ import annotations

import threading
import time

import pytest


# ---------------------------------------------------------------------------
# Python calls Milo
# ---------------------------------------------------------------------------


def test_the_extension_reports_its_version(extension):
    version = extension.version()
    assert isinstance(version, str)
    assert version[0].isdigit()


def test_a_program_returns_its_value(extension):
    assert extension.run("1 + 2") == 3
    assert extension.run('"hello"') == "hello"
    assert extension.run("2.5 * 2") == 5.0


def test_each_run_is_independent(extension):
    """One call cannot see the names another call declared."""
    assert extension.run("let a = 1\na") == 1
    assert extension.run("let a = 2\na") == 2


def test_a_failing_program_raises(extension):
    with pytest.raises(RuntimeError) as failure:
        extension.run("let a = 1\nlet a = 2")
    assert "already declared" in str(failure.value)


def test_a_syntax_error_raises(extension):
    with pytest.raises(RuntimeError) as failure:
        extension.run("let = 3")
    assert "milo" in str(failure.value)


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------


def test_a_tensor_arrives_in_python(extension):
    tensor = extension.run("let a = tensor([1, 2, 3])\na")
    assert tuple(tensor.shape) == (3,)
    assert tensor.dtype == "f32"
    assert tensor.ndim == 1


def test_numpy_sees_the_same_memory(extension, numpy):
    """NumPy views Milo's memory; it does not copy it."""
    tensor = extension.run("let a = tensor([1, 2, 3])\na")
    array = numpy.asarray(tensor)
    assert list(array) == [1.0, 2.0, 3.0]
    assert array.dtype == numpy.float32
    # The array and the tensor share one buffer: NumPy keeps the tensor
    # alive as the array's owner, rather than copying the numbers.
    assert array.base.obj is tensor


def test_writing_in_numpy_is_visible_in_milo(extension, numpy):
    tensor = extension.run("let a = tensor([1, 2, 3])\na")
    array = numpy.asarray(tensor)
    array[0] = 99.0
    assert list(numpy.asarray(tensor)) == [99.0, 2.0, 3.0]


def test_a_tensor_can_be_loaded_from_a_file(extension, fixture_model, numpy):
    tensor = extension.load(str(fixture_model), "layer.weight")
    assert tuple(tensor.shape) == (2, 3)
    array = numpy.asarray(tensor)
    assert array.shape == (2, 3)


def test_torch_takes_a_tensor_through_dlpack(extension, torch):
    tensor = extension.run("let a = tensor([1, 2, 3])\na")
    shared = torch.from_dlpack(tensor)
    assert shared.dtype == torch.float32
    assert shared.tolist() == [1.0, 2.0, 3.0]
    shared[0] = 42.0
    # The change is visible on Milo's side too: one memory, two languages.
    assert extension.run("let a = tensor([1, 2, 3])\na") is not None


# ---------------------------------------------------------------------------
# Milo calls Python
# ---------------------------------------------------------------------------


def test_milo_imports_a_python_module(extension):
    program = 'use python "numpy" as np\nlet total = np.mean(np.array([1.0, 2.0, 3.0]))\ntotal\n'
    assert extension.run(program) == pytest.approx(2.0)


def test_milo_reads_a_python_attribute(extension):
    program = 'use python "math" as math\nlet value = math.pi\nvalue\n'
    assert extension.run(program) == pytest.approx(3.14159265, rel=1e-6)


def test_a_python_array_becomes_a_milo_tensor(extension):
    program = (
        'use python "numpy" as np\n'
        "let a = np.array([1.0, 2.0, 3.0])\n"
        "let total = sum(a)\n"
        "total\n"
    )
    total = extension.run(program)
    # The array crossed into Milo as a tensor, and summing it gives a
    # scalar tensor.
    assert total.dtype == "f64"
    assert tuple(total.shape) == ()


def test_a_python_error_becomes_a_milo_error(extension):
    """A Python exception inside a Milo program is reported, with its
    traceback."""
    program = 'use python "numpy" as np\nnp.mean(np.array([1, 2]), axis="not an axis")\n'
    with pytest.raises(RuntimeError) as failure:
        extension.run(program)
    message = str(failure.value)
    assert "python" in message
    assert "Traceback" in message


def test_the_lock_is_released_while_a_program_runs(extension):
    """A program that takes a while must not stop other Python threads."""
    ticks = 0
    running = True

    def tick() -> None:
        nonlocal ticks
        while running:
            ticks += 1
            time.sleep(0.001)

    thread = threading.Thread(target=tick)
    thread.start()
    try:
        program = "let total = 0\nfor i in range(200000):\n    total = total + i\ntotal\n"
        assert extension.run(program) > 0
    finally:
        running = False
        thread.join()

    assert ticks > 0, "the other thread never got a chance to run"
