#!/usr/bin/env python3
"""Merge multiple .tdleaf.bin files with count-weighted averaging.

Each weight in the output is the weighted average of the corresponding
weights across all input files, where the weight for each file's entry
is its update count (cnt).  For entries present in only one file, they
pass through unchanged.  For entries with zero count in all files, the
value is taken from the first file that contains it.

Usage:
    # Merge .tdleaf.bin files only:
    python3 merge_tdleaf.py file1.tdleaf.bin file2.tdleaf.bin -o merged

    # Merge and also produce a .nnue file from merged weights:
    python3 merge_tdleaf.py file1.tdleaf.bin file2.tdleaf.bin -o merged --baseline net.nnue

The -o argument is a filename base: produces <base>.tdleaf.bin always,
and <base>.nnue when --baseline is given.  The .nnue is constructed by
applying merged float weights (divided by TDLEAF_SCALE=128) on top of
the baseline network, requantizing to int8/int16/int32.

The output .tdleaf.bin is a valid v10 file that can be loaded by Leaf.
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path

# Constants matching nnue.cpp / nnue.h
TDLEAF_MAGIC   = 0x544D4C46  # "TMLF"
TDLEAF_VERSION = 10
TDLEAF_SCALE   = 128.0
FNV_OFFSET     = 0xcbf29ce484222325
FNV_PRIME      = 0x100000001b3

LAYER_STACKS = 8
L0_SIZE      = 16
L0_INPUT     = 1024
L1_SIZE      = 32
L1_PADDED    = 32
L2_PADDED    = 32
HALF_DIMS    = 1024
PSQT_BKTS    = 8
FT_INPUTS    = 22528

LEB128_MAGIC = b"COMPRESSED_LEB128"


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
    """One layer stack's FC weights, counts, Adam v (second moment), and Adam m (first moment)."""
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
        # Adam v (second moment) — v6+
        self.v_l0_bias   = np.zeros(L0_SIZE, dtype=np.float32)
        self.v_l0_weight = np.zeros(L0_SIZE * L0_INPUT, dtype=np.float32)
        self.v_l1_bias   = np.zeros(L1_SIZE, dtype=np.float32)
        self.v_l1_weight = np.zeros(L1_SIZE * L1_PADDED, dtype=np.float32)
        self.v_l2_bias   = np.zeros(1, dtype=np.float32)
        self.v_l2_weight = np.zeros(L2_PADDED, dtype=np.float32)
        # Adam m (first moment) — v7+
        self.m_l0_bias   = np.zeros(L0_SIZE, dtype=np.float32)
        self.m_l0_weight = np.zeros(L0_SIZE * L0_INPUT, dtype=np.float32)
        self.m_l1_bias   = np.zeros(L1_SIZE, dtype=np.float32)
        self.m_l1_weight = np.zeros(L1_SIZE * L1_PADDED, dtype=np.float32)
        self.m_l2_bias   = np.zeros(1, dtype=np.float32)
        self.m_l2_weight = np.zeros(L2_PADDED, dtype=np.float32)

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

    def read_v(self, f):
        """Read Adam v (second moment) arrays — v6+."""
        self.v_l0_bias   = read_f32_array(f, L0_SIZE)
        self.v_l0_weight = read_f32_array(f, L0_SIZE * L0_INPUT)
        self.v_l1_bias   = read_f32_array(f, L1_SIZE)
        self.v_l1_weight = read_f32_array(f, L1_SIZE * L1_PADDED)
        self.v_l2_bias   = read_f32_array(f, 1)
        self.v_l2_weight = read_f32_array(f, L2_PADDED)

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

    def write_v(self, f):
        """Write Adam v arrays — v6+."""
        write_f32_array(f, self.v_l0_bias)
        write_f32_array(f, self.v_l0_weight)
        write_f32_array(f, self.v_l1_bias)
        write_f32_array(f, self.v_l1_weight)
        write_f32_array(f, self.v_l2_bias)
        write_f32_array(f, self.v_l2_weight)

    def read_m(self, f):
        """Read Adam m (first moment) arrays — v7+."""
        self.m_l0_bias   = read_f32_array(f, L0_SIZE)
        self.m_l0_weight = read_f32_array(f, L0_SIZE * L0_INPUT)
        self.m_l1_bias   = read_f32_array(f, L1_SIZE)
        self.m_l1_weight = read_f32_array(f, L1_SIZE * L1_PADDED)
        self.m_l2_bias   = read_f32_array(f, 1)
        self.m_l2_weight = read_f32_array(f, L2_PADDED)

    def write_m(self, f):
        """Write Adam m arrays — v7+."""
        write_f32_array(f, self.m_l0_bias)
        write_f32_array(f, self.m_l0_weight)
        write_f32_array(f, self.m_l1_bias)
        write_f32_array(f, self.m_l1_weight)
        write_f32_array(f, self.m_l2_bias)
        write_f32_array(f, self.m_l2_weight)


class TDLeafFile:
    """Parsed .tdleaf.bin v10 file."""
    def __init__(self):
        self.version = TDLEAF_VERSION
        # FT content fingerprint of the source .nnue (v10+).  0 = legacy/unknown.
        self.nnue_content_hash = 0
        self.fc = [FCBlock() for _ in range(LAYER_STACKS)]
        # Sparse FT/PSQT: dict keyed by feature index
        #   ft_rows[fi] = (ft_w[HALF_DIMS], ft_c[HALF_DIMS],
        #                   psqt_w[PSQT_BKTS], psqt_c[PSQT_BKTS])
        self.ft_rows = {}
        # FT biases
        self.ft_bias_w = np.zeros(HALF_DIMS, dtype=np.float32)
        self.ft_bias_c = np.zeros(HALF_DIMS, dtype=np.uint32)
        # Dense piece values: 6 piece types (v9+; v5-v8 had 6 × 8 PSQT buckets)
        self.piece_val_w = np.zeros(6, dtype=np.float32)
        self.piece_val_c = np.zeros(6, dtype=np.uint32)
        # Adam v section (v6+)
        self.t_adam = 0
        self.v_ft_bias   = np.zeros(HALF_DIMS, dtype=np.float32)
        self.v_piece_val = np.zeros(6, dtype=np.float32)
        # Sparse PSQT v: dict keyed by feature index → v_psqt[PSQT_BKTS]
        self.psqt_v_rows = {}
        # Adam m section (v7+)
        self.m_ft_bias   = np.zeros(HALF_DIMS, dtype=np.float32)
        self.m_piece_val = np.zeros(6, dtype=np.float32)
        # Sparse PSQT m: dict keyed by feature index → m_psqt[PSQT_BKTS]
        self.psqt_m_rows = {}
        # Sparse FT v section (v8+): dict keyed by feature index → v_ft[HALF_DIMS]
        self.ft_v_rows = {}

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, 'rb') as f:
            magic = read_u32(f)
            version = read_u32(f)
            if magic != TDLEAF_MAGIC:
                raise ValueError(f"{path}: bad magic 0x{magic:08X} (expected 0x{TDLEAF_MAGIC:08X})")
            if version not in (2, 3, 4, 5, 6, 7, 8, 9, 10):
                raise ValueError(f"{path}: unsupported version {version}")
            obj.version = version

            # v10+: nnue_content_hash immediately after version.
            if version >= 10:
                obj.nnue_content_hash = read_u32(f)

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

            # Dense piece values (v5+).  v9+: float32[6] + uint32[6].
            # v5-v8: float32[6][8] + uint32[6][8] → collapse to [6] by averaging
            # across PSQT buckets (matches engine load semantics in
            # nnue_training.cpp).
            if version >= 5:
                if version >= 9:
                    obj.piece_val_w = read_f32_array(f, 6)
                    obj.piece_val_c = read_u32_array(f, 6)
                else:
                    pv_w_48 = read_f32_array(f, 6 * PSQT_BKTS).reshape(6, PSQT_BKTS)
                    pv_c_48 = read_u32_array(f, 6 * PSQT_BKTS).reshape(6, PSQT_BKTS)
                    obj.piece_val_w = (pv_w_48.sum(axis=1) / PSQT_BKTS).astype(np.float32)
                    obj.piece_val_c = (pv_c_48.sum(axis=1) // PSQT_BKTS).astype(np.uint32)

            if version >= 6:
                obj.t_adam = read_u32(f)
                for s in range(LAYER_STACKS):
                    obj.fc[s].read_v(f)
                obj.v_ft_bias = read_f32_array(f, HALF_DIMS)
                # v_piece_val: max-per-piece-type when collapsing v6-v8.
                if version >= 9:
                    obj.v_piece_val = read_f32_array(f, 6)
                else:
                    v_pv_48 = read_f32_array(f, 6 * PSQT_BKTS).reshape(6, PSQT_BKTS)
                    obj.v_piece_val = v_pv_48.max(axis=1).astype(np.float32)
                n_pv_rows = read_u32(f)
                for _ in range(n_pv_rows):
                    fi = read_u32(f)
                    if fi >= FT_INPUTS:
                        break
                    obj.psqt_v_rows[fi] = read_f32_array(f, PSQT_BKTS)

            if version >= 7:
                for s in range(LAYER_STACKS):
                    obj.fc[s].read_m(f)
                obj.m_ft_bias = read_f32_array(f, HALF_DIMS)
                # m_piece_val: mean-per-piece-type when collapsing v7-v8.
                if version >= 9:
                    obj.m_piece_val = read_f32_array(f, 6)
                else:
                    m_pv_48 = read_f32_array(f, 6 * PSQT_BKTS).reshape(6, PSQT_BKTS)
                    obj.m_piece_val = m_pv_48.mean(axis=1).astype(np.float32)
                n_pm_rows = read_u32(f)
                for _ in range(n_pm_rows):
                    fi = read_u32(f)
                    if fi >= FT_INPUTS:
                        break
                    obj.psqt_m_rows[fi] = read_f32_array(f, PSQT_BKTS)

            if version >= 8:
                n_ftv_rows = read_u32(f)
                for _ in range(n_ftv_rows):
                    fi = read_u32(f)
                    if fi >= FT_INPUTS:
                        break
                    obj.ft_v_rows[fi] = read_f32_array(f, HALF_DIMS)

        return obj

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<II', TDLEAF_MAGIC, TDLEAF_VERSION))
            # v10+: nnue_content_hash immediately after version.
            f.write(struct.pack('<I', self.nnue_content_hash & 0xFFFFFFFF))

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

            # FT biases (v4+)
            write_f32_array(f, self.ft_bias_w)
            write_u32_array(f, self.ft_bias_c)

            # Dense piece values (v9+: float32[6] + uint32[6])
            write_f32_array(f, self.piece_val_w)
            write_u32_array(f, self.piece_val_c)

            # Adam v section (v6)
            f.write(struct.pack('<I', self.t_adam))
            for s in range(LAYER_STACKS):
                self.fc[s].write_v(f)
            write_f32_array(f, self.v_ft_bias)
            write_f32_array(f, self.v_piece_val)
            # Sparse PSQT v — same dirty rows as FT weights
            sorted_pv_fi = sorted(self.psqt_v_rows.keys())
            f.write(struct.pack('<I', len(sorted_pv_fi)))
            for fi in sorted_pv_fi:
                f.write(struct.pack('<I', fi))
                write_f32_array(f, self.psqt_v_rows[fi])

            # Adam m section (v7)
            for s in range(LAYER_STACKS):
                self.fc[s].write_m(f)
            write_f32_array(f, self.m_ft_bias)
            write_f32_array(f, self.m_piece_val)
            # Sparse PSQT m
            sorted_pm_fi = sorted(self.psqt_m_rows.keys())
            f.write(struct.pack('<I', len(sorted_pm_fi)))
            for fi in sorted_pm_fi:
                f.write(struct.pack('<I', fi))
                write_f32_array(f, self.psqt_m_rows[fi])

            # Sparse FT v section (v8)
            sorted_ftv_fi = sorted(self.ft_v_rows.keys())
            f.write(struct.pack('<I', len(sorted_ftv_fi)))
            for fi in sorted_ftv_fi:
                f.write(struct.pack('<I', fi))
                write_f32_array(f, self.ft_v_rows[fi])


# ---------------------------------------------------------------------------
# .nnue file I/O
# ---------------------------------------------------------------------------

def sleb128_decode_i16(data, count):
    """Decode SLEB128-compressed int16 values."""
    # Decode into uint16 buffer first, then view as int16 (avoids overflow warnings)
    raw = np.zeros(count, dtype=np.uint16)
    pos = 0
    for i in range(count):
        val = 0
        shift = 0
        while True:
            byte = data[pos]; pos += 1
            val |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                break
        if shift < 32 and (byte & 0x40):
            val |= (~0) << shift
        raw[i] = val & 0xFFFF
    return raw.view(np.int16)


def sleb128_decode_i32(data, count):
    """Decode SLEB128-compressed int32 values."""
    raw = np.zeros(count, dtype=np.uint32)
    pos = 0
    for i in range(count):
        val = 0
        shift = 0
        while True:
            byte = data[pos]; pos += 1
            val |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                break
        if shift < 64 and (byte & 0x40):
            val |= (~0) << shift
        raw[i] = val & 0xFFFFFFFF
    return raw.view(np.int32)


def sleb128_encode_i16(values):
    """Encode int16 values as SLEB128 bytes."""
    out = bytearray()
    for v in values:
        val = int(np.int16(v))
        while True:
            byte = val & 0x7F
            val >>= 7  # arithmetic shift for signed
            more = not ((val == 0 and not (byte & 0x40)) or (val == -1 and (byte & 0x40)))
            if more:
                byte |= 0x80
            out.append(byte)
            if not more:
                break
    return bytes(out)


def sleb128_encode_i32(values):
    """Encode int32 values as SLEB128 bytes."""
    out = bytearray()
    for v in values:
        val = int(np.int32(v))
        while True:
            byte = val & 0x7F
            val >>= 7
            more = not ((val == 0 and not (byte & 0x40)) or (val == -1 and (byte & 0x40)))
            if more:
                byte |= 0x80
            out.append(byte)
            if not more:
                break
    return bytes(out)


def read_leb128_section(f, count, decoder):
    """Read a COMPRESSED_LEB128 section from a .nnue file."""
    magic = f.read(17)
    if magic == LEB128_MAGIC:
        nbytes = struct.unpack('<I', f.read(4))[0]
        data = f.read(nbytes)
        return decoder(data, count)
    else:
        # Uncompressed fallback
        f.seek(-17, 1)
        elem_size = 2 if decoder is sleb128_decode_i16 else 4
        raw = f.read(count * elem_size)
        dtype = '<i2' if elem_size == 2 else '<i4'
        return np.frombuffer(raw, dtype=dtype).copy()


def write_leb128_section(f, values, encoder):
    """Write a COMPRESSED_LEB128 section to a .nnue file."""
    f.write(LEB128_MAGIC)
    encoded = encoder(values)
    f.write(struct.pack('<I', len(encoded)))
    f.write(encoded)


class NNUEFile:
    """Parsed .nnue file (HalfKAv2_hm format)."""
    def __init__(self):
        self.version = 0
        self.file_hash = 0
        self.desc = b""
        self.ft_hash = 0
        self.ft_biases  = np.zeros(HALF_DIMS, dtype=np.int16)
        self.ft_weights = np.zeros(FT_INPUTS * HALF_DIMS, dtype=np.int16)
        self.psqt_weights = np.zeros(FT_INPUTS * PSQT_BKTS, dtype=np.int32)
        self.stack_hashes = np.zeros(LAYER_STACKS, dtype=np.uint32)
        # FC weights stored in output-major layout (file layout, not vdotq)
        self.l0_biases  = np.zeros((LAYER_STACKS, L0_SIZE), dtype=np.int32)
        self.l0_weights = np.zeros((LAYER_STACKS, L0_SIZE * L0_INPUT), dtype=np.int8)
        self.l1_biases  = np.zeros((LAYER_STACKS, L1_SIZE), dtype=np.int32)
        self.l1_weights = np.zeros((LAYER_STACKS, L1_SIZE * L1_PADDED), dtype=np.int8)
        self.l2_biases  = np.zeros(LAYER_STACKS, dtype=np.int32)
        self.l2_weights = np.zeros((LAYER_STACKS, L2_PADDED), dtype=np.int8)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, 'rb') as f:
            obj.version   = read_u32(f)
            obj.file_hash = read_u32(f)
            desc_size     = read_u32(f)
            obj.desc      = f.read(desc_size) if desc_size > 0 else b""
            obj.ft_hash   = read_u32(f)

            obj.ft_biases    = read_leb128_section(f, HALF_DIMS, sleb128_decode_i16)
            obj.ft_weights   = read_leb128_section(f, FT_INPUTS * HALF_DIMS, sleb128_decode_i16)
            obj.psqt_weights = read_leb128_section(f, FT_INPUTS * PSQT_BKTS, sleb128_decode_i32)

            for s in range(LAYER_STACKS):
                obj.stack_hashes[s] = read_u32(f)
                obj.l0_biases[s]  = np.frombuffer(f.read(L0_SIZE * 4), dtype='<i4').copy()
                obj.l0_weights[s] = np.frombuffer(f.read(L0_SIZE * L0_INPUT), dtype=np.int8).copy()
                obj.l1_biases[s]  = np.frombuffer(f.read(L1_SIZE * 4), dtype='<i4').copy()
                obj.l1_weights[s] = np.frombuffer(f.read(L1_SIZE * L1_PADDED), dtype=np.int8).copy()
                obj.l2_biases[s]  = struct.unpack('<i', f.read(4))[0]
                obj.l2_weights[s] = np.frombuffer(f.read(L2_PADDED), dtype=np.int8).copy()

        return obj

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', self.version))
            f.write(struct.pack('<I', self.file_hash))
            f.write(struct.pack('<I', len(self.desc)))
            if self.desc:
                f.write(self.desc)
            f.write(struct.pack('<I', self.ft_hash))

            write_leb128_section(f, self.ft_biases, sleb128_encode_i16)
            print("  FT weights (23M values, may take a moment)...")
            write_leb128_section(f, self.ft_weights, sleb128_encode_i16)
            write_leb128_section(f, self.psqt_weights, sleb128_encode_i32)

            for s in range(LAYER_STACKS):
                f.write(struct.pack('<I', int(self.stack_hashes[s])))
                f.write(self.l0_biases[s].astype('<i4').tobytes())
                f.write(self.l0_weights[s].astype(np.int8).tobytes())
                f.write(self.l1_biases[s].astype('<i4').tobytes())
                f.write(self.l1_weights[s].astype(np.int8).tobytes())
                f.write(struct.pack('<i', int(self.l2_biases[s])))
                f.write(self.l2_weights[s].astype(np.int8).tobytes())


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

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


def fnv1a_u32_ft_hash(ft_weights_i16):
    """FNV-1a hash matching engine nnue_fnv1a_u32 over the FT weight bytes.

    Mirrors nnue.cpp:nnue_fnv1a_u32 — 64-bit FNV-1a processed 8 bytes at a
    time, then folded to 32 bits with (h >> 32) ^ (uint32_t)h.
    """
    data = np.ascontiguousarray(ft_weights_i16, dtype='<i2').tobytes()
    n8 = len(data) // 8
    h = FNV_OFFSET
    if n8:
        words = np.frombuffer(data[:n8 * 8], dtype='<u8')
        for w in words:
            h = ((h ^ int(w)) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    for b in data[n8 * 8:]:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return ((h >> 32) ^ (h & 0xFFFFFFFF)) & 0xFFFFFFFF


def merge_files(files, output_base, baseline_path=None, report=False):
    """Load all files, merge with count-weighting, write output."""
    loaded = []
    for path in files:
        print(f"Loading {path} ...", end=' ')
        td = TDLeafFile.load(path)
        hash_str = f"0x{td.nnue_content_hash:08X}" if td.version >= 10 else "n/a"
        print(f"v{td.version}, {len(td.ft_rows)} FT rows, .nnue hash={hash_str}")
        loaded.append(td)

    if not loaded:
        print("No files to merge.", file=sys.stderr)
        return

    # Sanity-check that all v10+ inputs reference the same source .nnue.
    v10_hashes = {td.nnue_content_hash for td in loaded if td.version >= 10}
    if len(v10_hashes) > 1:
        print(f"WARNING: input files reference different .nnue content hashes: "
              f"{sorted(f'0x{h:08X}' for h in v10_hashes)}.  "
              f"Merging anyway, but trained deltas may be inconsistent.",
              file=sys.stderr)

    out = TDLeafFile()
    # Default to the first v10 hash; will be overwritten below if --baseline
    # is given (we then recompute the hash from the output .nnue's FT weights).
    if v10_hashes:
        out.nnue_content_hash = next(iter(v10_hashes))

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

    # --- Dense piece values (size 6, v9+) ---
    # Clamp ≥ 0: matches engine guard against negative piece values (which
    # invert material evaluation and cause an unrecoverable death spiral).
    pv_pairs = [(td.piece_val_w, td.piece_val_c) for td in loaded]
    out.piece_val_w, out.piece_val_c = weighted_merge_arrays(pv_pairs)
    out.piece_val_w = np.maximum(out.piece_val_w, 0.0).astype(np.float32)

    # --- Adam v (max-merge) ---
    out.t_adam = max(td.t_adam for td in loaded)
    for s in range(LAYER_STACKS):
        out.fc[s].v_l0_bias   = np.maximum.reduce([td.fc[s].v_l0_bias   for td in loaded])
        out.fc[s].v_l0_weight = np.maximum.reduce([td.fc[s].v_l0_weight for td in loaded])
        out.fc[s].v_l1_bias   = np.maximum.reduce([td.fc[s].v_l1_bias   for td in loaded])
        out.fc[s].v_l1_weight = np.maximum.reduce([td.fc[s].v_l1_weight for td in loaded])
        out.fc[s].v_l2_bias   = np.maximum.reduce([td.fc[s].v_l2_bias   for td in loaded])
        out.fc[s].v_l2_weight = np.maximum.reduce([td.fc[s].v_l2_weight for td in loaded])
    out.v_ft_bias   = np.maximum.reduce([td.v_ft_bias   for td in loaded])
    out.v_piece_val = np.maximum.reduce([td.v_piece_val for td in loaded])
    # Sparse PSQT v: max-merge per dirty row
    all_pv_fi = set()
    for td in loaded:
        all_pv_fi.update(td.psqt_v_rows.keys())
    zero_pv = np.zeros(PSQT_BKTS, dtype=np.float32)
    for fi in sorted(all_pv_fi):
        vals = [td.psqt_v_rows.get(fi, zero_pv) for td in loaded]
        out.psqt_v_rows[fi] = np.maximum.reduce(vals)

    # --- Adam m (mean-merge) ---
    for s in range(LAYER_STACKS):
        out.fc[s].m_l0_bias   = np.mean([td.fc[s].m_l0_bias   for td in loaded], axis=0)
        out.fc[s].m_l0_weight = np.mean([td.fc[s].m_l0_weight for td in loaded], axis=0)
        out.fc[s].m_l1_bias   = np.mean([td.fc[s].m_l1_bias   for td in loaded], axis=0)
        out.fc[s].m_l1_weight = np.mean([td.fc[s].m_l1_weight for td in loaded], axis=0)
        out.fc[s].m_l2_bias   = np.mean([td.fc[s].m_l2_bias   for td in loaded], axis=0)
        out.fc[s].m_l2_weight = np.mean([td.fc[s].m_l2_weight for td in loaded], axis=0)
    out.m_ft_bias   = np.mean([td.m_ft_bias   for td in loaded], axis=0)
    out.m_piece_val = np.mean([td.m_piece_val for td in loaded], axis=0)
    # Sparse PSQT m: mean-merge per dirty row
    all_pm_fi = set()
    for td in loaded:
        all_pm_fi.update(td.psqt_m_rows.keys())
    zero_pm = np.zeros(PSQT_BKTS, dtype=np.float32)
    for fi in sorted(all_pm_fi):
        present = [td.psqt_m_rows[fi] for td in loaded if fi in td.psqt_m_rows]
        out.psqt_m_rows[fi] = np.mean(present, axis=0).astype(np.float32)

    # --- Sparse FT v (max-merge) — v8 ---
    all_ftv_fi = set()
    for td in loaded:
        all_ftv_fi.update(td.ft_v_rows.keys())
    zero_ftv = np.zeros(HALF_DIMS, dtype=np.float32)
    for fi in sorted(all_ftv_fi):
        vals = [td.ft_v_rows.get(fi, zero_ftv) for td in loaded]
        out.ft_v_rows[fi] = np.maximum.reduce(vals)

    # --- Write .nnue if baseline provided.  This runs before the .tdleaf.bin
    # save so we can stamp the OUTPUT .nnue's FT content hash into the
    # .tdleaf.bin (matching the engine's load-time pairing check).
    if baseline_path:
        out_nnue = write_merged_nnue(out, baseline_path, output_base + '.nnue')
        out.nnue_content_hash = fnv1a_u32_ft_hash(out_nnue.ft_weights)
        print(f"  Stamped .tdleaf.bin with output .nnue FT hash="
              f"0x{out.nnue_content_hash:08X}")

    # --- Write .tdleaf.bin ---
    tdleaf_path = output_base + '.tdleaf.bin'
    out.save(tdleaf_path)
    print(f"\nWrote {tdleaf_path}: {len(out.ft_rows)} FT rows, "
          f".nnue hash=0x{out.nnue_content_hash:08X}")

    if report:
        print_report(loaded, out)

    return out


def write_merged_nnue(merged_td, baseline_path, nnue_out_path):
    """Apply merged .tdleaf.bin weights onto a baseline .nnue and write output.

    Float weights from .tdleaf.bin are at TDLEAF_SCALE; divide by 128 to get
    raw float values, then requantize to int8/int16/int32 matching nnue_requantize_fc().
    Features not present in the .tdleaf.bin keep their baseline values.
    """
    print(f"\nLoading baseline {baseline_path} ...")
    baseline = NNUEFile.load(baseline_path)
    out = NNUEFile()

    # Copy header
    out.version      = baseline.version
    out.file_hash    = baseline.file_hash
    out.ft_hash      = baseline.ft_hash
    out.stack_hashes = baseline.stack_hashes.copy()

    # Update description
    orig_desc = baseline.desc.decode('utf-8', errors='replace') if baseline.desc else ""
    if orig_desc and "Trained by Leaf" not in orig_desc:
        new_desc = f"{orig_desc} Trained by Leaf TDLeaf"
    elif orig_desc:
        new_desc = orig_desc
    else:
        new_desc = "Trained by Leaf TDLeaf"
    out.desc = new_desc.encode('utf-8')

    # --- FC layers: tdleaf float / SCALE → requantize ---
    for s in range(LAYER_STACKS):
        b = merged_td.fc[s]

        # FC0 biases: float at SCALE → int32
        f32 = b.l0_bias_w / TDLEAF_SCALE
        out.l0_biases[s] = np.clip(np.round(f32), -2147483647, 2147483647).astype(np.int32)

        # FC0 weights: float at SCALE → int8 [-127, 127]
        # .tdleaf.bin stores in natural output-major layout [o * L0_INPUT + i],
        # same as the .nnue file layout (no vdotq reordering).
        f32 = b.l0_weight_w / TDLEAF_SCALE
        out.l0_weights[s] = np.clip(np.round(f32), -127, 127).astype(np.int8)

        # FC1 biases
        f32 = b.l1_bias_w / TDLEAF_SCALE
        out.l1_biases[s] = np.clip(np.round(f32), -2147483647, 2147483647).astype(np.int32)

        # FC1 weights
        f32 = b.l1_weight_w / TDLEAF_SCALE
        out.l1_weights[s] = np.clip(np.round(f32), -127, 127).astype(np.int8)

        # FC2 bias
        f32 = b.l2_bias_w[0] / TDLEAF_SCALE
        out.l2_biases[s] = np.int32(np.clip(np.round(f32), -2147483647, 2147483647))

        # FC2 weights
        f32 = b.l2_weight_w / TDLEAF_SCALE
        out.l2_weights[s] = np.clip(np.round(f32), -127, 127).astype(np.int8)

    # --- FT biases: override from tdleaf if any updates, else keep baseline ---
    if np.any(merged_td.ft_bias_c > 0):
        f32 = merged_td.ft_bias_w / TDLEAF_SCALE
        out.ft_biases = np.clip(np.round(f32), -32767, 32767).astype(np.int16)
    else:
        out.ft_biases = baseline.ft_biases.copy()

    # --- FT weights + PSQT: override only features present in tdleaf ---
    out.ft_weights   = baseline.ft_weights.copy()
    out.psqt_weights = baseline.psqt_weights.copy()

    n_updated = 0
    for fi, (ft_w, ft_c, ps_w, ps_c) in merged_td.ft_rows.items():
        if fi >= FT_INPUTS:
            continue
        # FT weights: float at SCALE → int16
        f32 = ft_w / TDLEAF_SCALE
        start = fi * HALF_DIMS
        out.ft_weights[start:start + HALF_DIMS] = np.clip(
            np.round(f32), -32767, 32767).astype(np.int16)
        # PSQT: float at SCALE → int32
        f32p = ps_w / TDLEAF_SCALE
        pstart = fi * PSQT_BKTS
        out.psqt_weights[pstart:pstart + PSQT_BKTS] = np.round(f32p).astype(np.int32)
        n_updated += 1

    print(f"Applied {n_updated} FT rows from .tdleaf.bin onto baseline")
    print(f"Writing {nnue_out_path} ...")
    out.save(nnue_out_path)
    print(f"Wrote {nnue_out_path}")
    return out


def print_report(loaded, merged):
    """Print summary statistics about the merge (output is v10)."""
    print("\n--- Merge Report ---")
    print(f"Input files: {len(loaded)}")
    print(f"Output format: v10 (.nnue content hash header + dense piece_val[6] "
          f"+ Adam m first-moment + sparse FT v second-moment)")

    # FC total counts per file
    for i, td in enumerate(loaded):
        total = sum(int(td.fc[s].l0_weight_c.sum()) +
                    int(td.fc[s].l1_weight_c.sum()) +
                    int(td.fc[s].l2_weight_c.sum())
                    for s in range(LAYER_STACKS))
        ft_total = sum(int(row[1].sum()) for row in td.ft_rows.values())
        pv_total = int(td.piece_val_c.sum())
        print(f"  File {i}: FC weight updates={total:,}, "
              f"FT rows={len(td.ft_rows):,}, FT weight updates={ft_total:,}, "
              f"piece_val updates={pv_total:,}")

    # Merged stats
    m_total = sum(int(merged.fc[s].l0_weight_c.sum()) +
                  int(merged.fc[s].l1_weight_c.sum()) +
                  int(merged.fc[s].l2_weight_c.sum())
                  for s in range(LAYER_STACKS))
    m_ft_total = sum(int(row[1].sum()) for row in merged.ft_rows.values())
    m_pv_total = int(merged.piece_val_c.sum())
    print(f"  Merged: FC weight updates={m_total:,}, "
          f"FT rows={len(merged.ft_rows):,}, FT weight updates={m_ft_total:,}, "
          f"piece_val updates={m_pv_total:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple .tdleaf.bin files with count-weighted averaging.")
    parser.add_argument('files', nargs='+', help='.tdleaf.bin input files')
    parser.add_argument('-o', '--output', required=True,
                        help='output filename base (produces <base>.tdleaf.bin and optionally <base>.nnue)')
    parser.add_argument('--baseline', default=None,
                        help='.nnue baseline file; when given, also produces <output>.nnue')
    parser.add_argument('--report', action='store_true', help='print merge statistics')
    args = parser.parse_args()

    for path in args.files:
        if not Path(path).exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    if args.baseline and not Path(args.baseline).exists():
        print(f"Error: baseline {args.baseline} not found", file=sys.stderr)
        sys.exit(1)

    merge_files(args.files, args.output, baseline_path=args.baseline, report=args.report)


if __name__ == '__main__':
    main()
