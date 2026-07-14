#!/usr/bin/env python3
"""
migrate_legacy_work.py — one-time backlog migration for iterations produced
before train.py's per-run archive model existed (the pre-migration
hybrid_loop.py, and any early train.py runs predating this script).  Applies
the same retention rules train.py now uses going forward, and backfills a
<tag>_final.json sidecar so --continue works on this net for the next
iteration.

Per --tags entry, in the order given (each tag's parent = the previous tag
in the list):
  - reconstructs games_this_iter (from the online-generation PGN's game
    count), depth (from that PGN's filename), epochs and picked_epoch (from
    the snapshots still in <tag>_work/train/, matched against <tag>_final.*
    by content), epoch_ladder and final_gauntlet results (by re-scoring the
    still-present PGNs the same way train.py does), and gauntlet_anchors
    (from the final-gauntlet PGN opponent names, unless --transcript gives
    something more precise);
  - moves the online-generation and final-gauntlet PGNs (previously flat in
    learn/pgn/<netbase>/ and learn/) into <tag>_work/, matching where
    train.py writes them today, then reuses train.py's own prune_work_dir()
    to prune raw dumps / non-winning epoch .nnue / epoch-ladder PGNs and
    gzip corpus.tsv + the online PGN;
  - deletes the scattered per-epoch/pretrain rating binaries left flat in
    learn/ and run/ (single-use, already rated), and the two backup
    checkpoint files superseded by <tag>_final.tdleaf.bin;
  - writes <tag>_final.json.

bt_lr/bt_lambda/bt_K/bt_td_lambda are NOT recoverable from on-disk artifacts
(the orchestrator's own hyperparameter log line went to the terminal, not to
any file) — they stay null in the reconstructed sidecar unless --transcript
points at a saved copy of the original invocations, in which case they're
parsed from the actual `--bt-lr`/etc. flags used.

DRY RUN BY DEFAULT.  Prints every action without touching anything.  Read
the output, then re-run with --apply to actually execute.  This operates on
irreplaceable training data — always dry-run first.

Usage (from engine/learn/):
    python3 ../scripts/migrate_legacy_work.py \\
        --tags material_260708-1e5g material_260708-5e5g \\
               material_260708-1e6g material_260708-2e6g material_260708-3e6g
    # review the output, then:
    python3 ../scripts/migrate_legacy_work.py --tags ... --apply

    # optionally, to recover bt_lr/bt_lambda/bt_K/bt_td_lambda precisely:
    python3 ../scripts/migrate_legacy_work.py --tags ... \\
        --transcript /path/to/saved_terminal_transcript.txt --apply
"""

import argparse
import filecmp
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # reuse LEARN_DIR/RUN_DIR, pgn_score, gzip_and_remove, prune_work_dir, log


def parse_transcript(path):
    """Extract per-tag hyperparameters from a saved copy of the original
    `train.py --tag ...` / `hybrid_loop.py --tag ...` invocations.  Returns
    {tag: {flag_name: value}}.  Best-effort regex scan, not a shell parser —
    only handles the simple `--flag value` invocations these commands
    actually use."""
    text = Path(path).read_text(errors="replace")
    out = {}
    for cmd in re.findall(
            r'(?:hybrid_loop|train)\.py\s+(.+?)(?=\n\S|\n\n|\Z)', text, re.S):
        cmd = " ".join(cmd.split())  # collapse continuation newlines/backslashes
        m = re.search(r'--tag\s+(\S+)', cmd)
        if not m:
            continue
        tag = m.group(1)
        fields = {}
        for flag, key, caster in (
                ("--depth", "depth", int),
                ("--bt-lr", "bt_lr", float),
                ("--bt-lambda", "bt_lambda", float),
                ("--bt-K", "bt_K", float),
                ("--bt-td-lambda", "bt_td_lambda", float),
                ("--games", "games_this_iter", int)):
            fm = re.search(rf'{re.escape(flag)}\s+(\S+)', cmd)
            if fm:
                try:
                    fields[key] = caster(fm.group(1))
                except ValueError:
                    pass
        gm = re.search(r'--gauntlet-anchors\s+((?:\S+\s+)*?)(?=--|\Z)', cmd)
        if gm:
            fields["gauntlet_anchors"] = gm.group(1).split()
        out[tag] = fields
    return out


def count_pgn_games(pgn_path):
    text = Path(pgn_path).read_text(errors="replace")
    return len(re.findall(r'\[Result "', text))


def find_online_pgn(tag):
    """Old layout: learn/pgn/<netbase>/match_<tag>_d<depth>.pgn."""
    matches = sorted(T.LEARN_DIR.glob(f"pgn/*/match_{tag}_d*.pgn"))
    return matches[0] if matches else None


def find_final_gauntlet_pgns(tag):
    return sorted(T.LEARN_DIR.glob(f"match_{tag}-final_vs_*.pgn"))


def find_epoch_ladder_pgns(tag):
    return sorted(T.LEARN_DIR.glob(f"match_{tag}-ep*_vs_*.pgn"))


def find_scattered_epoch_binaries(tag):
    """Per-epoch/pretrain rating binaries + their .nnue companions, left flat
    in both learn/ and run/ by the pre-migration script."""
    out = []
    for base in (T.LEARN_DIR, T.RUN_DIR):
        out += sorted(base.glob(f"Leaf_v{tag}-ep*"))
        out += sorted(base.glob(f"Leaf_v{tag}-pretrain*"))
    return out


def find_stray_run_final_binary(tag):
    """The pre-migration script left Leaf_v<tag>-final resident in run/ too
    (only the staged .nnue was cleaned up) — the learn/ copy is the one
    that's kept; the run/ copy is a stray duplicate."""
    return sorted(T.RUN_DIR.glob(f"Leaf_v{tag}-final*"))


def reconstruct_epoch_ladder(tag, actions):
    results = []
    for pgn in find_epoch_ladder_pgns(tag):
        m = re.search(rf'match_{re.escape(tag)}-ep(\d+)_vs_(.+)\.pgn$', pgn.name)
        if not m:
            actions.append(f"  WARNING: unrecognized epoch-ladder PGN name {pgn.name} — skipped")
            continue
        ep = int(m.group(1))
        W, L, D, elo, err = T.pgn_score(pgn, f"{tag}-ep{ep}")
        results.append((ep, (W, L, D, elo, err)))
    results.sort(key=lambda r: r[0])
    return results


def reconstruct_final_gauntlet(tag, actions):
    results = []
    anchors = []
    for pgn in find_final_gauntlet_pgns(tag):
        m = re.search(rf'match_{re.escape(tag)}-final_vs_(.+)\.pgn$', pgn.name)
        if not m:
            actions.append(f"  WARNING: unrecognized final-gauntlet PGN name {pgn.name} — skipped")
            continue
        opp = "Leaf_v" + m.group(1)
        anchors.append(opp)
        results.append((opp, T.pgn_score(pgn, f"{tag}-final")))
    return results, anchors


def find_picked_epoch(tag, tdir, final_nnue, actions):
    if not tdir.is_dir():
        actions.append(f"  WARNING: {tdir} missing — can't determine picked_epoch")
        return None, None
    epochs = sorted(int(m.group(1)) for f in tdir.glob(f"{tag}_ep*.nnue")
                    if (m := re.match(rf'{re.escape(tag)}_ep(\d+)\.nnue$', f.name)))
    if not epochs:
        actions.append(f"  WARNING: no {tag}_ep*.nnue snapshots in {tdir} — can't determine picked_epoch")
        return None, len(epochs)
    for ep in epochs:
        cand = tdir / f"{tag}_ep{ep}.nnue"
        if final_nnue.is_file() and filecmp.cmp(cand, final_nnue, shallow=False):
            return ep, len(epochs)
    actions.append(f"  WARNING: no epoch snapshot in {tdir} byte-matches "
                   f"{final_nnue.name} — picked_epoch left null")
    return None, len(epochs)


def find_base_net(tag, tdir, actions):
    if not tdir.is_dir():
        return None
    candidates = [f for f in tdir.glob("*.nnue")
                 if not re.match(rf'{re.escape(tag)}_ep\d+\.nnue$', f.name)
                 and f.name != f"{tag}_pretrain.nnue"]
    if len(candidates) == 1:
        return candidates[0].name
    if candidates:
        actions.append(f"  WARNING: ambiguous base-net candidates in {tdir}: "
                       f"{[c.name for c in candidates]} — net left null")
    return None


def migrate_tag(tag, parent_tag, cumulative_so_far, transcript_data,
                apply, keep_epoch_states, actions):
    work = T.LEARN_DIR / f"{tag}_work"
    tdir = work / "train"
    final_nnue = T.LEARN_DIR / f"{tag}_final.nnue"
    final_td = T.LEARN_DIR / f"{tag}_final.tdleaf.bin"
    sidecar_path = T.LEARN_DIR / f"{tag}_final.json"
    netbase_for_backups = None

    actions.append(f"\n=== {tag} ===")
    if not final_nnue.is_file() or not final_td.is_file():
        actions.append(f"  SKIP: missing {final_nnue.name}/{final_td.name} "
                       "— not a completed run, leaving untouched")
        return None
    if sidecar_path.is_file():
        actions.append(f"  SKIP: {sidecar_path.name} already exists (already migrated)")
        return json.loads(sidecar_path.read_text())

    tx = transcript_data.get(tag, {})

    net = tx.get("net") or find_base_net(tag, tdir, actions) or None
    if net:
        netbase_for_backups = net[:-5] if net.endswith(".nnue") else net
        actions.append(f"  net (reconstructed): {net}")
    else:
        actions.append("  WARNING: could not determine base net — "
                       "'net' left null, fix the sidecar by hand before --continue")

    online_pgn = find_online_pgn(tag)
    depth = tx.get("depth")
    games_this_iter = tx.get("games_this_iter")
    if online_pgn:
        dm = re.search(r'_d(\d+)\.pgn$', online_pgn.name)
        if depth is None and dm:
            depth = int(dm.group(1))
        if games_this_iter is None:
            games_this_iter = count_pgn_games(online_pgn)
        actions.append(f"  online PGN: {online_pgn.relative_to(T.LEARN_DIR)} "
                       f"({games_this_iter} games, depth {depth})")
    else:
        actions.append("  WARNING: no online-generation PGN found — "
                       "games_this_iter/depth left null (--skip-online run?)")

    pick_ep, epochs = find_picked_epoch(tag, tdir, final_nnue, actions)

    epoch_ladder = reconstruct_epoch_ladder(tag, actions)
    ladder_pgns = find_epoch_ladder_pgns(tag)
    if epoch_ladder:
        actions.append(f"  epoch ladder reconstructed from {len(ladder_pgns)} PGN(s): "
                       + ", ".join(f"ep{ep} Elo{elo:+.0f}" for ep, (_, _, _, elo, _) in epoch_ladder))

    final_gauntlet, pgn_anchors = reconstruct_final_gauntlet(tag, actions)
    gauntlet_pgns = find_final_gauntlet_pgns(tag)
    if final_gauntlet:
        actions.append(f"  final gauntlet reconstructed from {len(gauntlet_pgns)} PGN(s): "
                       + ", ".join(f"{opp} Elo{elo:+.0f}" for opp, (_, _, _, elo, _) in final_gauntlet))

    gauntlet_anchors = tx.get("gauntlet_anchors")
    if gauntlet_anchors is None:
        gauntlet_anchors = pgn_anchors
        if pgn_anchors:
            actions.append(f"  gauntlet_anchors (from final-gauntlet PGN opponents): {pgn_anchors}")

    cumulative_games = cumulative_so_far + (games_this_iter or 0)

    sidecar = {
        "tag": tag,
        "net": net,
        "parent_tag": parent_tag,
        "date": time.strftime("%Y-%m-%d", time.localtime(
            work.stat().st_mtime if work.exists() else time.time())),
        "games_this_iter": games_this_iter,
        "cumulative_games": cumulative_games,
        "depth": depth,
        "epochs": epochs,
        "picked_epoch": pick_ep,
        "bt_lr": tx.get("bt_lr"),
        "bt_lambda": tx.get("bt_lambda"),
        "bt_K": tx.get("bt_K"),
        "bt_td_lambda": tx.get("bt_td_lambda"),
        "gauntlet_anchors": gauntlet_anchors,
        "epoch_ladder": [
            {"epoch": ep, "W": W, "L": L, "D": D, "elo": elo, "err": err}
            for ep, (W, L, D, elo, err) in epoch_ladder
        ],
        "final_gauntlet": [
            {"opponent": opp, "W": W, "L": L, "D": D, "elo": elo, "err": err}
            for opp, (W, L, D, elo, err) in final_gauntlet
        ],
        "_reconstructed": True,
        "_reconstructed_note": "backfilled by migrate_legacy_work.py from "
                               "on-disk artifacts — bt_lr/bt_lambda/bt_K/"
                               "bt_td_lambda are only accurate if --transcript "
                               "covered this tag; verify before relying on them.",
    }

    # ---- append the reconstructed tables to train.log, matching what a live
    # run writes (train.py's main() does this before its own pruning sweep).
    if tdir.is_dir() and (tdir / "train.log").is_file():
        log_chunks = []
        if epoch_ladder:
            ladder_opp_name = f"{tag}-pretrain"
            log_chunks.append("\n" + "\n".join(T.render_epoch_ladder(
                tag, ladder_opp_name, "?", "?", epoch_ladder)) + "\n")
        if final_gauntlet:
            log_chunks.append("\n" + "\n".join(T.render_gauntlet(
                f"Leaf_v{tag}-final", final_gauntlet)) + "\n")
        if log_chunks:
            actions.append(f"  APPEND reconstructed epoch-ladder/gauntlet tables to "
                           f"{(tdir / 'train.log').relative_to(T.LEARN_DIR)}")
            if apply:
                with open(tdir / "train.log", "a") as f:
                    f.write("\n[migrate_legacy_work.py: tables below are "
                           "reconstructed from PGNs, not captured live]\n")
                    for chunk in log_chunks:
                        f.write(chunk)

    # ---- reorganize: move kept PGNs into work/, matching where train.py
    # writes them today, so train.py's own prune_work_dir() can find and
    # prune the epoch-ladder ones and gzip the rest exactly as it would for
    # a live run.
    if work.is_dir():
        if online_pgn and online_pgn.is_file():
            dest = work / f"match_{tag}_d{depth}.pgn"
            actions.append(f"  MOVE {online_pgn.relative_to(T.LEARN_DIR)} -> {dest.relative_to(T.LEARN_DIR)}")
            if apply:
                online_pgn.rename(dest)
        for pgn in gauntlet_pgns:
            dest = work / pgn.name
            actions.append(f"  MOVE {pgn.relative_to(T.LEARN_DIR)} -> {dest.relative_to(T.LEARN_DIR)}")
            if apply:
                pgn.rename(dest)
        for pgn in ladder_pgns:
            actions.append(f"  DELETE {pgn.relative_to(T.LEARN_DIR)} (epoch-ladder, already scored above)")
            if apply:
                pgn.unlink()

        actions.append(f"  PRUNE {work.relative_to(T.LEARN_DIR)} "
                       "(raw dumps, non-winning epoch .nnue"
                       + ("" if keep_epoch_states else ", per-epoch .tdleaf.bin")
                       + "; gzip corpus.tsv + online PGN)")
        if apply:
            T.prune_work_dir(work, tdir, work / "epoch_binaries", tag,
                             pick_ep if pick_ep is not None else (epochs or 0),
                             keep_epoch_states)
    else:
        actions.append(f"  WARNING: {work} does not exist — nothing to prune/reorganize")

    for f in find_scattered_epoch_binaries(tag):
        actions.append(f"  DELETE {f.relative_to(T.ENGINE_DIR)} (single-use, already rated)")
        if apply:
            f.unlink()
    for f in find_stray_run_final_binary(tag):
        actions.append(f"  DELETE {f.relative_to(T.ENGINE_DIR)} (stray duplicate — "
                       "the kept copy is learn/" + f"Leaf_v{tag}-final)")
        if apply:
            f.unlink()

    if netbase_for_backups:
        for name in (f"{netbase_for_backups}.tdleaf.bin-pre{tag}",
                    f"{netbase_for_backups}.tdleaf.bin-{tag}-online"):
            stray = T.LEARN_DIR / name
            if stray.is_file():
                actions.append(f"  DELETE {name} (superseded by {tag}_final.tdleaf.bin)")
                if apply:
                    stray.unlink()

    actions.append(f"  WRITE {sidecar_path.name}: {json.dumps(sidecar, indent=None)[:200]}...")
    if apply:
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

    return sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True,
                    help="Tags in chain order (each tag's parent = the previous one)")
    ap.add_argument("--transcript", default=None,
                    help="Saved copy of the original invocations, to recover "
                         "bt_lr/bt_lambda/bt_K/bt_td_lambda/gauntlet_anchors precisely")
    ap.add_argument("--keep-epoch-states", action="store_true",
                    help="Keep every epoch's .tdleaf.bin (same meaning as train.py's flag)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually perform the migration (default: dry run only)")
    args = ap.parse_args()

    if Path.cwd().resolve() != T.LEARN_DIR.resolve():
        T.die(f"run from {T.LEARN_DIR} (cwd is {Path.cwd()})")

    transcript_data = parse_transcript(args.transcript) if args.transcript else {}

    if not args.apply:
        print("=== DRY RUN — no files will be touched. Pass --apply to execute. ===")

    actions = []
    cumulative = 0
    parent = None
    for tag in args.tags:
        sidecar = migrate_tag(tag, parent, cumulative, transcript_data,
                              args.apply, args.keep_epoch_states, actions)
        if sidecar is not None:
            cumulative = sidecar["cumulative_games"]
            parent = tag

    for line in actions:
        print(line)

    if not args.apply:
        print("\n=== DRY RUN complete. Review the actions above, then re-run "
             "with --apply. ===")
    else:
        print("\n=== Migration applied. ===")


if __name__ == "__main__":
    main()
