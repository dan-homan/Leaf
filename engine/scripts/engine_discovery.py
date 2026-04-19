#!/usr/bin/env python3
#
# Shared engine discovery / resolution / interactive picker used by match.py
# and training_run.py.
#
# Conventions:
#   - Leaf binaries live in engine/run/ and are named Leaf_v<version>
#   - External engines live in tools/engines/<name>/ with an executable file.
#   - All paths returned are absolute.
#

import os
import stat


script_dir = os.path.dirname(os.path.abspath(__file__))
run_dir    = os.path.normpath(os.path.join(script_dir, "../run"))
learn_dir  = os.path.normpath(os.path.join(script_dir, "../learn"))
tools_dir  = os.path.normpath(os.path.join(script_dir, "../../tools"))


def _is_executable(path):
    try:
        return bool(os.stat(path).st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except OSError:
        return False


def resolve_exe(name):
    """Return absolute path: check cwd, run_dir, and learn_dir (in that order)."""
    if os.path.isabs(name):
        return name
    if os.path.isfile(name):
        return os.path.abspath(name)
    for base in (run_dir, learn_dir):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return os.path.join(run_dir, name)   # fallback (existence checked by caller)


def discover_leaf_engines():
    """Return [(name, abs_path), ...] of executables in run/ named Leaf_v*."""
    out = []
    if not os.path.isdir(run_dir):
        return out
    for f in sorted(os.listdir(run_dir)):
        if not f.startswith("Leaf_v"):
            continue
        if f.endswith((".lock", ".py", ".pl", ".txt")):
            continue
        p = os.path.join(run_dir, f)
        if os.path.isfile(p) and _is_executable(p):
            out.append((f, p))
    return out


def discover_external_engines():
    """Return [(dir_name, abs_path), ...] of engines under tools/engines/<name>/.

    For each engine directory, pick the executable whose filename contains the
    directory name; if none match, pick the largest executable.
    """
    out = []
    engines_dir = os.path.join(tools_dir, "engines")
    if not os.path.isdir(engines_dir):
        return out
    for d in sorted(os.listdir(engines_dir)):
        dp = os.path.join(engines_dir, d)
        if not os.path.isdir(dp):
            continue
        candidates = []
        for f in os.listdir(dp):
            fp = os.path.join(dp, f)
            if os.path.isfile(fp) and _is_executable(fp):
                try:
                    candidates.append((f, fp, os.stat(fp).st_size))
                except OSError:
                    pass
        if not candidates:
            continue
        named = [c for c in candidates if d.lower() in c[0].lower()]
        best  = max(named or candidates, key=lambda c: c[2])
        out.append((d, best[1]))
    return out


def discover_engines():
    """Return (leaf_engines, external_engines) tuple — see individual helpers."""
    return discover_leaf_engines(), discover_external_engines()


def pick_engine(prompt, leaf_engines=None, external_engines=None,
                allow_custom=True):
    """Interactive engine picker.  Returns (display_name, abs_path).

    `display_name` is the basename (for Leaf) or directory name (for externals).
    If `allow_custom` is True, the user may type 'c' to enter an arbitrary path.
    """
    if leaf_engines is None or external_engines is None:
        leaf_engines, external_engines = discover_engines()

    print(f"\n{prompt}")
    entries = []   # (display_name, abs_path)
    idx = 1
    if leaf_engines:
        print("  Leaf binaries (engine/run/):")
        for name, path in leaf_engines:
            print(f"    [{idx}] {name}")
            entries.append((name, path))
            idx += 1
    if external_engines:
        print("  External engines (tools/engines/):")
        for name, path in external_engines:
            print(f"    [{idx}] {name}")
            entries.append((name, path))
            idx += 1
    if allow_custom:
        print("  [c] Custom path")
    print()

    while True:
        choice = input("  Select: ").strip()
        if not choice:
            continue
        if allow_custom and choice.lower() == "c":
            while True:
                p = input("  Path to engine: ").strip()
                if not p:
                    break
                if os.path.isfile(p):
                    return (os.path.basename(p), os.path.abspath(p))
                rp = os.path.join(run_dir, p)
                if os.path.isfile(rp):
                    return (os.path.basename(p), os.path.abspath(rp))
                lp = os.path.join(learn_dir, p)
                if os.path.isfile(lp):
                    return (os.path.basename(p), os.path.abspath(lp))
                print(f"  File not found: {p}")
            continue
        try:
            n = int(choice)
            if 1 <= n <= len(entries):
                return entries[n - 1]
        except ValueError:
            p = resolve_exe(choice)
            if os.path.isfile(p):
                return (choice, p)
        print(f"  Invalid selection: {choice}")
