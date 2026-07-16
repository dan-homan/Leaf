#!/usr/bin/env python3
"""Does offline consolidation of the SEED on the hybrid corpus move the seed
toward the displaced generator (post-online net)?

A = post_online - seed   (online displacement of the corpus generator)
B = ep1 - seed           (what consolidation did to the seed on that corpus)
C = ep2 - ep1            (second epoch movement)

Reports cos(A,B) and proj of B onto A (fraction of the generator displacement
replicated) per section, plus PSQT per bucket.
"""
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location(
    "cnl", "/Users/homand/Leaf/engine/scripts/compare_nnue_learning.py")
cnl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnl)

L = "/Users/homand/Leaf/engine/learn/"
seed = cnl.read_tdleaf_fc(L + "material_260708-5e6g_final.tdleaf.bin")
gen  = cnl.read_tdleaf_fc(L + "material_260708-6e6g_work/train/nn-material_260708.tdleaf.bin")
ep1  = cnl.read_tdleaf_fc(L + "seedctl-260716_work/train/seedctl-260716_ep1.tdleaf.bin")
ep2  = cnl.read_tdleaf_fc(L + "seedctl-260716_work/train/seedctl-260716_ep2.tdleaf.bin")


def stats(name, A, B, C):
    A = A.astype(np.float64).ravel(); B = B.astype(np.float64).ravel()
    C = C.astype(np.float64).ravel()
    nA = np.linalg.norm(A); nB = np.linalg.norm(B); nC = np.linalg.norm(C)
    cosAB = float(A @ B) / (nA * nB + 1e-12)
    projAB = float(A @ B) / (nA * nA + 1e-12)   # how much of A is inside B
    cosAC = float(A @ C) / (nA * nC + 1e-12)
    print(f"  {name:14s} |A|={nA:9.0f} |B|={nB:9.0f} |C|={nC:9.0f}  "
          f"cos(A,B)={cosAB:+.2f}  proj B on A={projAB:+.2f}  cos(A,C)={cosAC:+.2f}")


print("A = generator online displacement (seed->post-online hybrid)")
print("B = seedctl ep1 movement (seed->ep1)   C = ep2-ep1")
print()
for key in ['fc0_bias', 'fc0_w', 'fc1_bias', 'fc1_w', 'fc2_bias', 'fc2_w']:
    A = np.concatenate([np.asarray(x, np.float64).ravel() for x in gen[key]]) - \
        np.concatenate([np.asarray(x, np.float64).ravel() for x in seed[key]])
    B = np.concatenate([np.asarray(x, np.float64).ravel() for x in ep1[key]]) - \
        np.concatenate([np.asarray(x, np.float64).ravel() for x in seed[key]])
    C = np.concatenate([np.asarray(x, np.float64).ravel() for x in ep2[key]]) - \
        np.concatenate([np.asarray(x, np.float64).ravel() for x in ep1[key]])
    stats(key, A, B, C)

# PSQT + FT on common rows
maps = []
for d in (seed, gen, ep1, ep2):
    maps.append({fi: k for k, fi in enumerate(d['ft_fi'])})
fis = [fi for fi in seed['ft_fi'] if all(fi in m for m in maps[1:])]
idx = [np.array([m[fi] for fi in fis]) for m in maps]
ps = [d['psqt_w'][i] for d, i in zip((seed, gen, ep1, ep2), idx)]
ft = [d['ft_w'][i] for d, i in zip((seed, gen, ep1, ep2), idx)]

stats('ft_w', ft[1] - ft[0], ft[2] - ft[0], ft[3] - ft[2])
stats('psqt(all)', ps[1] - ps[0], ps[2] - ps[0], ps[3] - ps[2])
print("\nPSQT per bucket:")
for b in range(8):
    stats(f"psqt b{b}", ps[1][:, b] - ps[0][:, b],
          ps[2][:, b] - ps[0][:, b], ps[3][:, b] - ps[2][:, b])
