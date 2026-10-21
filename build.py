#!/usr/bin/env python3
"""Milo's one top-level build script.

Drives both toolchains:

  * CMake  -> the C/C++ runtime, the `milo` CLI, and the test suite
  * Cargo  -> the Rust `mpm` package manager (and anything else in the
              Cargo workspace)

Examples
--------
    python build.py                      # debug build + run the test suite
    python build.py --release            # optimised build
    python build.py --release --no-tests # optimised build, skip ctest
    python build.py --clean              # wipe build/ and start over
    python build.py --install            # install into build/<preset>/install
    python build.py --no-rust            # C/C++ only
    python build.py --preset relwithdebinfo

The script is deliberately dependency-free: it uses only the Python standard
library so that a fresh clone can be built with nothing but a compiler,
CMake, and Cargo already installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_ROOT = ROOT / "build"

PRESETS = ("debug", "release", "relwithdebinfo")


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def run(cmd, *, cwd=ROOT, env=None, check=True, capture=False, echo=True):
    """Run a command, streaming or capturing its output."""
    printable = " ".join(str(c) for c in cmd)
    if echo and not capture:
        print(f"\033[36m$\033[0m {printable}", flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        env=env,
        check=False,
        text=True,
        capture_output=capture,
    )
    if check and proc.returncode != 0:
        if capture:
            sys.stdout.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
        print(f"\033[31merror:\033[0m command failed ({proc.returncode}): {printable}",
              file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


# ---------------------------------------------------------------------------
# Windows: MSVC environment discovery
# ---------------------------------------------------------------------------


def _find_vcvars() -> Path | None:
    """Locate vcvars64.bat inside a Visual Studio installation."""
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Microsoft Visual Studio",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        # Layout: <root>/<year>/<edition>/VC/Auxiliary/Build/vcvars64.bat
        for year in sorted(root.glob("*")):
            for edition in sorted(year.glob("*")):
                vcvars = edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if vcvars.is_file():
                    return vcvars
    return None


def _find_msvc_tools() -> tuple[Path, Path, str] | None:
    """Find the MSVC toolset and the Windows SDK.

    Returns (vc_tools_dir, sdk_root, sdk_version) or None.
    """
    vs_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio"
    vc_tools = None
    if vs_root.is_dir():
        for year in sorted(vs_root.glob("*")):
            for edition in sorted(year.glob("*")):
                tools = edition / "VC" / "Tools" / "MSVC"
                if not tools.is_dir():
                    continue
                versions = sorted((p for p in tools.glob("*") if p.is_dir()),
                                  key=lambda p: p.name)
                if versions:
                    vc_tools = versions[-1]

    sdk_root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10"
    sdk = None
    sdk_version = None
    if sdk_root.is_dir():
        versions = sorted((p for p in (sdk_root / "Include").glob("*") if p.is_dir()),
                          key=lambda p: p.name)
        if versions:
            sdk_version = versions[-1].name
            sdk = sdk_root

    if vc_tools is None or sdk is None:
        return None
    return vc_tools, sdk, sdk_version


def _manual_msvc_env(env: dict) -> dict:
    """Build an MSVC environment without vcvars.

    Some Visual Studio installations have a broken `vcvars64.bat` (for
    example when only part of the C++ workload is present). The toolset and
    the Windows SDK are still on disk, so we can point the compiler at them
    directly.
    """
    found = _find_msvc_tools()
    if found is None:
        return env
    vc_tools, sdk, sdk_version = found
    sdk_include = sdk / "Include" / sdk_version
    sdk_lib = sdk / "Lib" / sdk_version
    sdk_bin = sdk / "bin" / sdk_version / "x64"

    include_dirs = [
        vc_tools / "include",
        sdk_include / "ucrt",
        sdk_include / "um",
        sdk_include / "shared",
        sdk_include / "winrt",
        sdk_include / "cppwinrt",
    ]
    lib_dirs = [
        vc_tools / "lib" / "x64",
        sdk_lib / "ucrt" / "x64",
        sdk_lib / "um" / "x64",
    ]
    bin_dirs = [vc_tools / "bin" / "Hostx64" / "x64", sdk_bin]

    env["INCLUDE"] = ";".join(str(p) for p in include_dirs if p.is_dir())
    env["LIB"] = ";".join(str(p) for p in lib_dirs if p.is_dir())
    path = env.get("PATH") or env.get("Path") or ""
    parts = [str(p) for p in bin_dirs if p.is_dir()]
    env["PATH"] = ";".join(parts + [path]) if path else ";".join(parts)
    return env


def ensure_msvc_env() -> dict:
    """Return an environment in which the MSVC compiler is usable.

    On Windows the Visual Studio toolchain is not on PATH by default; we
    import it by running vcvars64.bat and capturing its environment. If
    that fails we fall back to pointing the compiler at the toolset and the
    Windows SDK by hand.
    """
    env = dict(os.environ)
    if not sys.platform.startswith("win"):
        return env
    if have("cl"):
        return env

    vcvars = _find_vcvars()
    if vcvars is not None:
        print(f"\033[36m$\033[0m importing MSVC environment from {vcvars}")
        cmd = f'call "{vcvars}" >nul && set'
        proc = subprocess.run(
            ["cmd", "/c", cmd], capture_output=True, text=True, env=env, check=False
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key] = value
            if _on_path(env, "cl.exe"):
                return env

    env = _manual_msvc_env(env)
    if _on_path(env, "cl.exe"):
        print("\033[36m$\033[0m using the MSVC toolset directly (vcvars64.bat was unusable)")
        return env

    print(
        "\033[33mwarning:\033[0m MSVC not found; falling back to whatever "
        "compiler is on PATH.",
        file=sys.stderr,
    )
    return env


# ---------------------------------------------------------------------------
# CMake
# ---------------------------------------------------------------------------


def _on_path(env, tool: str) -> bool:
    """Is `tool` on the PATH in `env`?"""
    path = env.get("PATH") or env.get("Path") or ""
    for directory in path.split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, tool)
        if os.path.isfile(candidate) and (
            tool.lower().endswith(".exe") or os.access(candidate, os.X_OK)
        ):
            return True
    return False


def cmake_generator(env) -> list[str]:
    """Pick a generator.

    Preference order:
      1. Ninja + MSVC   (fast, on Windows)
      2. Ninja + the default compiler
      3. Visual Studio (multi-config)
      4. Unix Makefiles
    """
    if have("ninja"):
        return ["-G", "Ninja"]
    if sys.platform.startswith("win"):
        return ["-G", "Visual Studio 17 2022", "-A", "x64"]
    return ["-G", "Unix Makefiles"]


def configure(args, env) -> Path:
    build_dir = BUILD_ROOT / args.preset
    build_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["cmake", "-S", ROOT, "-B", build_dir, f"-DCMAKE_BUILD_TYPE={args.preset.capitalize()}"]
    cmd += cmake_generator(env)

    if args.preset == "debug" and not args.no_sanitizers:
        cmd.append("-DMILO_ENABLE_SANITIZERS=ON")
    else:
        cmd.append("-DMILO_ENABLE_SANITIZERS=OFF")

    if args.werror:
        cmd.append("-DMILO_WERROR=ON")
    if args.no_tests:
        cmd.append("-DMILO_BUILD_TESTS=OFF")
    if args.no_cli:
        cmd.append("-DMILO_BUILD_CLI=OFF")

    run(cmd, env=env)
    return build_dir


def python_directory(build_dir: Path) -> Path | None:
    """The directory of the Python that the C++ build found.

    The Milo runtime embeds Python, so anything that loads the runtime
    needs that Python's directory on the path.
    """
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Python3_EXECUTABLE:FILEPATH="):
            return Path(line.split("=", 1)[1]).parent
    return None


def build(build_dir: Path, env, jobs: int | None) -> None:
    config = "Debug" if env.get("MILO_PRESET", "debug") == "debug" else "Release"
    cmd = ["cmake", "--build", build_dir, "--config", config, "--parallel"]
    if jobs:
        cmd.append(str(jobs))
    run(cmd, env=env)


def copy_sanitizer_runtime(build_dir: Path, env) -> None:
    """Copy the sanitizer runtime next to the built binaries.

    MSVC's AddressSanitizer is a DLL that lives in the Visual Studio
    toolset directory. Without it the executables fail to start with
    STATUS_DLL_NOT_FOUND once you leave the developer prompt, so we copy it
    next to the binaries.
    """
    if sys.platform.startswith("win"):
        patterns = ("clang_rt.asan*.dll", "clang_rt.ubsan*.dll", "libasan*.dll",
                    "libubsan*.dll")
    else:
        return

    path = env.get("PATH") or env.get("Path") or ""
    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        return

    copied = 0
    for directory in path.split(os.pathsep):
        if not directory or not os.path.isdir(directory):
            continue
        for pattern in patterns:
            for dll in Path(directory).glob(pattern):
                target = bin_dir / dll.name
                if not target.exists():
                    try:
                        shutil.copy2(dll, target)
                        copied += 1
                    except OSError:
                        pass
    if copied:
        print(f"\033[32mcopied\033[0m {copied} sanitizer runtime file(s) into {bin_dir}")


def test(build_dir: Path, env) -> None:
    cmd = ["ctest", "--test-dir", build_dir, "--output-on-failure"]
    run(cmd, env=env)


def install(build_dir: Path, env) -> None:
    dest = build_dir / "install"
    run(["cmake", "--install", build_dir, "--prefix", dest], env=env)
    print(f"\033[32minstalled to\033[0m {dest}")


# ---------------------------------------------------------------------------
# Cargo
# ---------------------------------------------------------------------------


def cargo_build(args, env) -> None:
    if not have("cargo"):
        print(
            "\033[33mwarning:\033[0m cargo not found; skipping the Rust "
            "workspace (mpm).",
            file=sys.stderr,
        )
        return
    cmd = ["cargo", "build", "--manifest-path", ROOT / "Cargo.toml"]
    if args.preset == "release":
        cmd.append("--release")
    cmd += ["--target-dir", BUILD_ROOT / "cargo"]
    if args.jobs:
        cmd += ["-j", str(args.jobs)]
    run(cmd, env=env)

def build_docs() -> None:
    """Build the documentation website, if Node.js is available."""
    site = ROOT / "site"
    if not (site / "package.json").is_file():
        print(f"\033[33mwarning:\033[0m no documentation site at {site}")
        return
    npm = shutil.which("npm")
    if not npm:
        print("\033[33mwarning:\033[0m npm is not available; skipping the documentation site")
        return

    print("\033[36m$\033[0m building the documentation site")
    run([npm, "install", "--no-audit", "--no-fund"], cwd=site)
    run([npm, "run", "build"], cwd=site)
    print(f"\033[32mdocumentation built\033[0m {site / 'dist'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build Milo (C/C++ via CMake, Rust via Cargo).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--preset",
        choices=PRESETS,
        default="debug",
        help="build configuration (default: debug, which enables sanitizers)",
    )
    p.add_argument("--release", action="store_true", help="shorthand for --preset release")
    p.add_argument("--clean", action="store_true", help="remove the build directory first")
    p.add_argument("--test", action="store_true", help="run the test suite (default: on for debug)")
    p.add_argument("--no-tests", action="store_true", help="do not build or run the test suite")
    p.add_argument("--install", action="store_true", help="install into build/<preset>/install")
    p.add_argument("--no-rust", action="store_true", help="skip the Cargo workspace")
    p.add_argument("--no-cli", action="store_true", help="skip the `milo` executable")
    p.add_argument("--werror", action="store_true", help="treat warnings as errors")
    p.add_argument(
        "--no-sanitizers",
        action="store_true",
        help="build without AddressSanitizer/UBSan (some toolchains do not ship them)",
    )
    p.add_argument("--jobs", "-j", type=int, default=None, help="parallel build jobs")
    p.add_argument(
        "--docs",
        action="store_true",
        help="also build the documentation website (needs Node.js)",
    )
    p.add_argument("--configure-only", action="store_true", help="stop after configuring")
    args = p.parse_args()

    if args.release:
        args.preset = "release"
    if args.no_tests:
        args.test = False
    elif args.preset == "debug":
        args.test = True

    if args.clean and BUILD_ROOT.exists():
        print(f"\033[36m$\033[0m removing {BUILD_ROOT}")
        shutil.rmtree(BUILD_ROOT)

    env = ensure_msvc_env()
    env["MILO_PRESET"] = args.preset

    if not have("cmake"):
        print("\033[31merror:\033[0m cmake is required but was not found on PATH.",
              file=sys.stderr)
        return 1

    build_dir = configure(args, env)
    if args.configure_only:
        return 0

    build(build_dir, env, args.jobs)
    copy_sanitizer_runtime(build_dir, env)

    # The Milo runtime embeds Python, so anything that loads the runtime
    # needs that Python's directory on the path.
    python_dir = python_directory(build_dir)
    if python_dir is not None:
        env["PATH"] = str(python_dir) + os.pathsep + env.get("PATH", "")

    if not args.no_rust:
        cargo_build(args, env)

    if args.test:
        test(build_dir, env)

    if args.install:
        install(build_dir, env)

    if args.docs:
        build_docs()

    exe = build_dir / "bin" / ("milo.exe" if sys.platform.startswith("win") else "milo")
    if exe.exists():
        print(f"\033[32mbuilt\033[0m {exe}")
        print(f"try:  {exe} repl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
