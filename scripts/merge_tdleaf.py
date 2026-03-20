#!/usr/bin/env python3
"""Merge multiple .tdleaf.bin files with count-weighted averaging.

Each weight in the output is the weighted average of the corresponding
weights across all input files, where the weight for each file's entry
is its update count (cnt).  For entries present in only one file, they
pass through unchanged.  For entries with zero count in all files, the
value is taken from the first file that contains it.

Usage:
    python3 merge_tdleaf.py file1.tdleaf.bin file2.tdleaf.bin [...] -o merged.tdleaf.bin

The output is a valid v4 .tdleaf.bin file that can be loaded by Leaf.
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path

# Constants matching nnue.cpp / nnue.h
TDLEAF_MAGIC   = 0x544D4C46  # "TMLF"
TDLEAF_VERSION = 4
TDLEAF_SCALE   = 128.0

LAYER_STACKS = 8
L0_SIZE      = 16
L0_INPUT     = 1024
L1_SIZE      = 32
L1_PADDED    = 32
L2_PADDED    = 32
HALF_DIMS    = 1024
PSQT_BKTS    = 8
FT_INPUTS    = 22528


def read_u32(f):
    data = f.read(4)
    if len(data) < 4:
        raise EOFError("unexpected end of file")
    return struct.unpack('<I', data)[0]


def read_f32_array(f, n):
    data = f.read(n * 4)
    if len(data) < n * 4:
        raise EOFError("unexpected end of file")
    return np.frombuffer(data, dtype='<f4').copy()


def read_u32_array(f, n):
    data = f.read(n * 4)
    if len(data) < n * 4:
        raise EOFError("unexpected end of file")
    return np.frombuffer(data, dtype='<u4').copy()


def write_f32_array(f, arr):
    f.write(arr.astype('<f4').tobytes())


def write_u32_array(f, arr):
    f.write(arr.astype('<u4').tobytes())


class FCBlock:
    """One layer stack's FC weights and counts."""
    def __init__(self):
        self.l0_bias_w   = np.zeros(L0_SIZE, dtype=np.float32)
        self.l0_bias_c   = np.zeros(L0_SIZE, dtype=np.uint32)
        self.l0_weight_w = np.zeros(L0_SIZE * L0_INPUT, dtype=np.float32)
        self.l0_weight_c = np.zeros(L0_SIZE * L0_INPUT, dtype=np.uint32)
        self.l1_bias_w   = np.zeros(L1_SIZE, dtype=np.float32)
        self.l1_bias_c   = np.zeros(L1_SIZE, dtype=np.uint32)
        self.l1_weight_w = np.zeros(L1_SIZE * L1_PADDED, dtype=np.float32)
        self.l1_weight_c = np.zeros(L1_SIZE * L1_PADDED, dtype=np.uint32)
        self.l2_bias_w   = np.zeros(1, dtype=np.float32)
        self.l2_bias_c   = np.zeros(1, dtype=np.uint32)
        self.l2_weight_w = np.zeros(L2_PADDED, dtype=np.float32)
        self.l2_weight_c = np.zeros(L2_PADDED, dtype=np.uint32)

    def read(self, f):
        self.l0_bias_w   = read_f32_array(f, L0_SIZE)
        self.l0_bias_c   = read_u32_array(f, L0_SIZE)
        self.l0_weight_w = read_f32_array(f, L0_SIZE * L0_INPUT)
        self.l0_weight_c = read_u32_array(f, L0_SIZE * L0_INPUT)
        self.l1_bias_w   = read_f32_array(f, L1_SIZE)
        self.l1_bias_c   = read_u32_array(f, L1_SIZE)
        self.l1_weight_w = read_f32_array(f, L1_SIZE * L1_PADDED)
        self.l1_weight_c = read_u32_array(f, L1_SIZE * L1_PADDED)
        self.l2_bias_w   = read_f32_array(f, 1)
        self.l2_bias_c   = read_u32_array(f, 1)
        self.l2_weight_w = read_f32_array(f, L2_PADDED)
        self.l2_weight_c = read_u32_array(f, L2_PADDED)

    def write(self, f):
        write_f32_array(f, self.l0_bias_w)
        write_u32_array(f, self.l0_bias_c)
        write_f32_array(f, self.l0_weight_w)
        write_u32_array(f, self.l0_weight_c)
        write_f32_array(f, self.l1_bias_w)
        write_u32_array(f, self.l1_bias_c)
        write_f32_array(f, self.l1_weight_w)
        write_u32_array(f, self.l1_weight_c)
        write_f32_array(f, self.l2_bias_w)
        write_u32_array(f, self.l2_bias_c)
        write_f32_array(f, self.l2_weight_w)
        write_u32_array(f, self.l2_weight_c)


class TDLeafFile:
    """Parsed .tdleaf.bin v4 file."""
    def __init__(self):
        self.version = TDLEAF_VERSION
        self.fc = [FCBlock() for _ in range(LAYER_STACKS)]
        # Sparse FT/PSQT: dict keyed by feature index
        #   ft_rows[fi] = (ft_w[HALF_DIMS], ft_c[HALF_DIMS],
        #                   psqt_w[PSQT_BKTS], psqt_c[PSQT_BKTS])
        self.ft_rows = {}
        # FT biases
        self.ft_bias_w = np.zeros(HALF_DIMS, dtype=np.float32)
        self.ft_bias_c = np.zeros(HALF_DIMS, dtype=np.uint32)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, 'rb') as f:
            magic = read_u32(f)
            version = read_u32(f)
            if magic != TDLEAF_MAGIC:
                raise ValueError(f"{path}: bad magic 0x{magic:08X} (expected 0x{TDLEAF_MAGIC:08X})")
            if version not in (2, 3, 4):
                raise ValueError(f"{path}: unsupported version {version}")
            obj.version = version

            for s in range(LAYER_STACKS):
                obj.fc[s].read(f)

            if version >= 3:
                n_ft_rows = read_u32(f)
                for _ in range(n_ft_rows):
                    fi = read_u32(f)
                    if fi >= FT_INPUTS:
                        break
                    ft_w  = read_f32_array(f, HALF_DIMS)
                    ft_c  = read_u32_array(f, HALF_DIMS)
                    ps_w  = read_f32_array(f, PSQT_BKTS)
                    ps_c  = read_u32_array(f, PSQT_BKTS)
                    obj.ft_rows[fi] = (ft_w, ft_c, ps_w, ps_c)

            if version >= 4:
                obj.ft_bias_w = read_f32_array(f, HALF_DIMS)
                obj.ft_bias_c = read_u32_array(f, HALF_DIMS)

        return obj

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<II', TDLEAF_MAGIC, TDLEAF_VERSION))

            for s in range(LAYER_STACKS):
                self.fc[s].write(f)

            # Sparse FT/PSQT — sorted by feature index for determinism
            sorted_fi = sorted(self.ft_rows.keys())
            f.write(struct.pack('<I', len(sorted_fi)))
            for fi in sorted_fi:
                ft_w, ft_c, ps_w, ps_c = self.ft_rows[fi]
                f.write(struct.pack('<I', fi))
                write_f32_array(f, ft_w)
                write_u32_array(f, ft_c)
                write_f32_array(f, ps_w)
                write_u32_array(f, ps_c)

            # FT biases
            write_f32_array(f, self.ft_bias_w)
            write_u32_array(f, self.ft_bias_c)


def weighted_merge_arrays(pairs):
    """Merge (value_scaled, count) pairs with count-weighted averaging.

    pairs: list of (values_array, counts_array) where values are at TDLEAF_SCALE.
    Returns: (merged_values_at_scale, merged_counts).

    For each element i:
      if total_count[i] > 0:
        merged[i] = sum(val[i] * cnt[i]) / total_count[i]
      else:
        merged[i] = val[i] from first file (passthrough)
      merged_cnt[i] = sum(cnt[i])
    """
    n = pairs[0][0].shape[0]
    total_cnt = np.zeros(n, dtype=np.uint64)
    weighted_sum = np.zeros(n, dtype=np.float64)
    first_val = pairs[0][0].astype(np.float64)

    for vals, cnts in pairs:
        c = cnts.astype(np.uint64)
        total_cnt += c
        weighted_sum += vals.astype(np.float64) * c

    merged = np.where(total_cnt > 0,
                      weighted_sum / np.maximum(total_cnt, 1),
                      first_val)
    return merged.astype(np.float32), total_cnt.astype(np.uint32)


def merge_files(files, output_path, report=False):
    """Load all files, merge with count-weighting, write output."""
    loaded = []
    for path in files:
        print(f"Loading {path} ...", end=' ')
        td = TDLeafFile.load(path)
        print(f"v{td.version}, {len(td.ft_rows)} FT rows")
        loaded.append(td)

    if not loaded:
        print("No files to merge.", file=sys.stderr)
        return

    out = TDLeafFile()

    # --- FC layers ---
    for s in range(LAYER_STACKS):
        pairs_list = {
            'l0_bias':   [], 'l0_weight': [],
            'l1_bias':   [], 'l1_weight': [],
            'l2_bias':   [], 'l2_weight': [],
        }
        for td in loaded:
            b = td.fc[s]
            pairs_list['l0_bias'].append((b.l0_bias_w, b.l0_bias_c))
            pairs_list['l0_weight'].append((b.l0_weight_w, b.l0_weight_c))
            pairs_list['l1_bias'].append((b.l1_bias_w, b.l1_bias_c))
            pairs_list['l1_weight'].append((b.l1_weight_w, b.l1_weight_c))
            pairs_list['l2_bias'].append((b.l2_bias_w, b.l2_bias_c))
            pairs_list['l2_weight'].append((b.l2_weight_w, b.l2_weight_c))

        ob = out.fc[s]
        ob.l0_bias_w,   ob.l0_bias_c   = weighted_merge_arrays(pairs_list['l0_bias'])
        ob.l0_weight_w, ob.l0_weight_c = weighted_merge_arrays(pairs_list['l0_weight'])
        ob.l1_bias_w,   ob.l1_bias_c   = weighted_merge_arrays(pairs_list['l1_bias'])
        ob.l1_weight_w, ob.l1_weight_c = weighted_merge_arrays(pairs_list['l1_weight'])
        ob.l2_bias_w,   ob.l2_bias_c   = weighted_merge_arrays(pairs_list['l2_bias'])
        ob.l2_weight_w, ob.l2_weight_c = weighted_merge_arrays(pairs_list['l2_weight'])

    # --- Sparse FT/PSQT rows ---
    # Collect all feature indices across all files
    all_fi = set()
    for td in loaded:
        all_fi.update(td.ft_rows.keys())

    zero_ft_w = np.zeros(HALF_DIMS, dtype=np.float32)
    zero_ft_c = np.zeros(HALF_DIMS, dtype=np.uint32)
    zero_ps_w = np.zeros(PSQT_BKTS, dtype=np.float32)
    zero_ps_c = np.zeros(PSQT_BKTS, dtype=np.uint32)

    for fi in sorted(all_fi):
        ft_pairs = []
        ps_pairs = []
        for td in loaded:
            if fi in td.ft_rows:
                ft_w, ft_c, ps_w, ps_c = td.ft_rows[fi]
                ft_pairs.append((ft_w, ft_c))
                ps_pairs.append((ps_w, ps_c))
            else:
                ft_pairs.append((zero_ft_w, zero_ft_c))
                ps_pairs.append((zero_ps_w, zero_ps_c))

        m_ft_w, m_ft_c = weighted_merge_arrays(ft_pairs)
        m_ps_w, m_ps_c = weighted_merge_arrays(ps_pairs)
        out.ft_rows[fi] = (m_ft_w, m_ft_c, m_ps_w, m_ps_c)

    # --- FT biases ---
    bias_pairs = [(td.ft_bias_w, td.ft_bias_c) for td in loaded]
    out.ft_bias_w, out.ft_bias_c = weighted_merge_arrays(bias_pairs)

    # --- Write ---
    out.save(output_path)
    print(f"\nWrote {output_path}: {len(out.ft_rows)} FT rows")

    if report:
        print_report(loaded, out)


def print_report(loaded, merged):
    """Print summary statistics about the merge."""
    print("\n--- Merge Report ---")
    print(f"Input files: {len(loaded)}")

    # FC total counts per file
    for i, td in enumerate(loaded):
        total = sum(int(td.fc[s].l0_weight_c.sum()) +
                    int(td.fc[s].l1_weight_c.sum()) +
                    int(td.fc[s].l2_weight_c.sum())
                    for s in range(LAYER_STACKS))
        ft_total = sum(int(row[1].sum()) for row in td.ft_rows.values())
        print(f"  File {i}: FC weight updates={total:,}, "
              f"FT rows={len(td.ft_rows):,}, FT weight updates={ft_total:,}")

    # Merged stats
    m_total = sum(int(merged.fc[s].l0_weight_c.sum()) +
                  int(merged.fc[s].l1_weight_c.sum()) +
                  int(merged.fc[s].l2_weight_c.sum())
                  for s in range(LAYER_STACKS))
    m_ft_total = sum(int(row[1].sum()) for row in merged.ft_rows.values())
    print(f"  Merged: FC weight updates={m_total:,}, "
          f"FT rows={len(merged.ft_rows):,}, FT weight updates={m_ft_total:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple .tdleaf.bin files with count-weighted averaging.")
    parser.add_argument('files', nargs='+', help='.tdleaf.bin input files')
    parser.add_argument('-o', '--output', required=True, help='output .tdleaf.bin path')
    parser.add_argument('--report', action='store_true', help='print merge statistics')
    args = parser.parse_args()

    for path in args.files:
        if not Path(path).exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    merge_files(args.files, args.output, report=args.report)


if __name__ == '__main__':
    main()
